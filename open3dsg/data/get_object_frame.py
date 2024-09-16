# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os
import json
import pickle
import multiprocessing
from functools import partial
import trimesh
import cv2
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
from open3dsg.config.config import CONF
from open3dsg.util.util_misc import read_txt_to_list
# from segment_anything import sam_model_registry, SamPredictor

lock = multiprocessing.Lock()

use_sam = False
# sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth').to(device="cuda")
# sam.share_memory()
# sam_predictor = SamPredictor(sam)


def sam_selection(img, points):
    k_rounds = 10
    k_sample = 5
    score_prime = 0
    mask_prime = None
    sam_predictor.set_image(img)

    for i in range(k_rounds):
        sampled_points = points[np.random.choice(points.shape[0], min(k_sample, points.shape[0]), replace=False), :]
        with torch.no_grad():
            mask, score, logits = sam_predictor.predict(
                point_coords=sampled_points,
                point_labels=np.ones(sampled_points.shape[0]),
                multimask_output=False,
            )
        if score > score_prime:
            mask_prime = mask
            score_prime = score
    return mask_prime


def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]

    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(float(data[4].strip().split(' ')[-1]))
        m_Height = int(float(data[5].strip().split(' ')[-1]))
        m_Shift = int(float(data[6].strip().split(' ')[-1]))
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(float(data[11].strip().split(' ')[-1]))

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size
    )


def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose


def get_label(label_path):
    label_list = []
    with open(label_path, "r") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list


def read_pointcloud_R3SCAN(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join(CONF.PATH.R3SCAN_RAW, scan_id, 'labels.instances.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)
    labels = np.array(plydata.metadata['_ply_raw']['vertex']['data']['objectId'])

    return points, labels


def read_json(root, split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    selected_scans = selected_scans.union(read_txt_to_list(os.path.join(root, f'{split}_scans.txt')))
    with open(os.path.join(root, f"relationships_{split}.json"), "r") as read_file:
        data = json.load(read_file)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

####################################################


def read_scan_info_R3SCAN(scan_id, mode='depth'):
    scan_path = os.path.join(CONF.PATH.R3SCAN_RAW, scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode=mode)
    # mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'

    image_list, color_list, extrinsic_list, frame_paths = [], [], [], []

    for i in range(0, intrinsic_info['m_frames_size'], 10):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + 'depth.pgm')
        color_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + 'color.jpg')
        frame_paths.append("frame-%s." % str(i).zfill(6) + 'color.jpg')
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + "pose.txt")
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)

        color_list.append(np.array(plt.imread(color_path)))

        image_list.append(cv2.imread(frame_path, -1).reshape(-1))
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic)

    return np.array(image_list), np.array(color_list), np.array(extrinsic_list), intrinsic_info, frame_paths

####################################################


def scannet_get_instance_ply(plydata, segs, aggre):
    ''' map idx to segments '''
    seg_map = dict()
    for idx in range(len(segs['segIndices'])):
        seg = segs['segIndices'][idx]
        if seg in seg_map:
            seg_map[seg].append(idx)
        else:
            seg_map[seg] = [idx]

    ''' Group segments '''
    aggre_seg_map = dict()
    for segGroup in aggre['segGroups']:
        aggre_seg_map[segGroup['id']] = list()
        for seg in segGroup['segments']:
            aggre_seg_map[segGroup['id']].extend(seg_map[seg])
    assert (len(aggre_seg_map) == len(aggre['segGroups']))
    # print('num of aggre_seg_map:',len(aggre_seg_map))

    ''' Over write label to segments'''
    # labels = plydata.vertices[:,0] # wrong but work around
    try:
        labels = plydata.metadata['_ply_raw']['vertex']['data']['label']
    except:
        labels = plydata.elements[0]['label']

    instances = np.zeros_like(labels)
    colors = plydata.visual.vertex_colors
    used_vts = set()
    for seg, indices in aggre_seg_map.items():
        s = set(indices)
        if len(used_vts.intersection(s)) > 0:
            raise RuntimeError('duplicate vertex')
        used_vts.union(s)
        for idx in indices:
            instances[idx] = seg

    return plydata, instances


def load_scannet(pth_ply, pth_seg, pth_agg, verbose=False, random_color=False):
    ''' Load GT '''
    plydata = trimesh.load(pth_ply, process=False)
    num_verts = plydata.vertices.shape[0]
    if verbose:
        print('num of verts:', num_verts)

    ''' Load segment file'''
    with open(pth_seg) as f:
        segs = json.load(f)
    if verbose:
        print('len(aggre[\'segIndices\']):', len(segs['segIndices']))
    segment_ids = list(np.unique(np.array(segs['segIndices'])))  # get unique segment ids
    if verbose:
        print('num of unique ids:', len(segment_ids))

    ''' Load aggregation file'''
    with open(pth_agg) as f:
        aggre = json.load(f)

    plydata, instances = scannet_get_instance_ply(plydata, segs, aggre)
    labels = plydata.metadata['_ply_raw']['vertex']['data']['label'].flatten()

    return plydata, labels, instances


def read_pointcloud_scannet(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    base = os.path.join(CONF.PATH.SCANNET_RAW3D, scan_id)
    plydata, labels, instances = load_scannet(os.path.join(base, scan_id + '_vh_clean_2.labels.ply'), os.path.join(
        base, scan_id + '_vh_clean_2.0.010000.segs.json'), os.path.join(base, scan_id + '_vh_clean.aggregation.json'))
    points = np.array(plydata.vertices)
    return points, labels, instances


def read_scan_info_scannet(scan_id, mode='depth'):
    scan_path = os.path.join(CONF.PATH.SCANNET_RAW2D, scan_id)
    # sequence_path = os.path.join(scan_path, "color")
    # intrinsic_path = os.path.join(scan_path, "_info.txt")
    intrinsic_info = dict() # read_intrinsic(intrinsic_path, mode=mode)

    image_list, color_list, extrinsic_list = [], [], []
    imgs = [img for img in sorted(os.listdir(os.path.join(scan_path, 'depth')), key=lambda x: int(os.path.splitext(x)[0]))]
    colors = [img for img in sorted(os.listdir(os.path.join(scan_path, 'color')), key=lambda x: int(os.path.splitext(x)[0]))]
    poses = [pose for pose in sorted(os.listdir(os.path.join(scan_path, 'pose')), key=lambda x: int(os.path.splitext(x)[0]))]

    # pylint: disable=consider-using-enumerate
    # print(imgs,poses)
    for i in range(0, len(imgs)):
        frame_path = os.path.join(scan_path, 'depth', imgs[i])
        color_path = os.path.join(scan_path, 'color', colors[i])
        extrinsic_path = os.path.join(scan_path, 'pose', poses[i])
        # print(frame_path,extrinsic_path)
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)
        image_list.append(cv2.imread(frame_path, -1).reshape(-1))
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_list.append(color_img)

        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic)
    # pylint: enable=consider-using-enumerate

    return np.array(image_list), np.array(color_list), np.array(extrinsic_list), intrinsic_info, colors

#############################


def compute_mapping(world_to_camera, coords, depth, intrinsic, cut_bound, vis_thres, image_dim):
    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coords_new.shape[0] == 4, "[!] Shape error"

    p = np.matmul(world_to_camera, coords_new)
    p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
    p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
    z = p[2].copy()
    pi = np.round(p).astype(int)  # simply round the projected coordinates
    inside_mask = (pi[0] >= cut_bound) * (pi[1] >= cut_bound) \
        * (pi[0] < image_dim[0]-cut_bound) \
        * (pi[1] < image_dim[1]-cut_bound)
    depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
    occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                            - p[2][inside_mask]) <= \
        vis_thres * depth_cur

    inside_mask[inside_mask == True] = occlusion_mask
    mapping[0][inside_mask] = pi[1][inside_mask]
    mapping[1][inside_mask] = pi[0][inside_mask]
    mapping[2][inside_mask] = 1

    return mapping


def image_3d_mapping(scan, image_list, color_list, img_names, point_cloud, instances, extrinsics, intrinsics, instance_names, image_width, image_height, boarder_pixels=0, vis_tresh=0.05, scene_data=None):
    # dict(zip(list(instance_names.keys()),[[] for i in range(len(image_list))])) #dict with keys inst and then frame,points tuple
    object2frame = dict()

    squeezed_instances = instances.squeeze()
    image_dim = np.array([image_width, image_height])
    for i, (extrinsic, depth, color) in enumerate(zip(extrinsics, image_list, color_list)):
        world_to_camera = np.linalg.inv(extrinsic)
        depth = depth.reshape(image_dim[::-1])/1000
        for inst in instance_names.keys():
            locs_in = point_cloud[squeezed_instances == int(inst)]
            mapping = compute_mapping(world_to_camera, locs_in, depth, np.array(intrinsics), boarder_pixels, vis_tresh, image_dim).T
            homog_points = mapping[:, 2] == 1
            ratio = (homog_points).sum()/mapping.shape[0]
            pixels = mapping[:, -1].sum()
            if pixels > 12 and ((ratio > 0.3 or pixels > 80) or (instance_names[inst] in ['wall', 'floor'] and pixels > 80)):
                if inst not in object2frame:
                    object2frame[inst] = []
                obj_points = mapping[homog_points]
                unique_mapping = np.unique(mapping[homog_points][:, :2], axis=0).astype(np.uint16)
                if use_sam:
                    obj_mask = sam_selection(color, obj_points[:, :2][:, ::-1])
                    rows, cols = np.where(obj_mask.squeeze())
                    object2frame[inst].append((img_names[i], pixels, ratio, (np.min(cols), np.min(rows), np.max(
                        cols), np.max(rows)), unique_mapping))
                else:
                    object2frame[inst].append((img_names[i], pixels, ratio, (obj_points[:, 1].min(), obj_points[:, 0].min(
                    ), obj_points[:, 1].max(), obj_points[:, 0].max()), unique_mapping))

    return object2frame


def viz_sam(obj_points, color, obj_mask):
    f = plt.figure()
    ax = f.add_subplot(131)
    ax.imshow(color)
    ax = f.add_subplot(132)
    ax.imshow(color)
    ax.scatter(obj_points[:, 1], obj_points[:, 0])
    ax = f.add_subplot(133)
    ax.imshow(color)
    mask_color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = obj_mask.reshape(*obj_mask.shape[-2:], 1) * mask_color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.show()


def run(scan, scene_data, export_path, dataset):
    export_dir = export_path
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=False)
    # if not scan == "scene0358_00":
    #     return
    output_filepath = os.path.join(export_dir, f"{scan}_object2image.pkl")
    if os.path.exists(output_filepath):
        # print('path already exists: ', output_filepath)
        return
    instance_names = scene_data[scan]['obj']
    if dataset == "R3SCAN":
        pc_i, instances_i = read_pointcloud_R3SCAN(scan)
        image_list, color_list, extrinsic_list, intrinsic_info, img_names = read_scan_info_R3SCAN(scan)
    else:
        pc_i, labels_i, instances_i = read_pointcloud_scannet(scan)
        instance_names = dict(zip(instances_i, labels_i))
        image_list, color_list, extrinsic_list, intrinsic_info, img_names = read_scan_info_scannet(scan)
        intrinsic_info['m_intrinsic'] = np.loadtxt(os.path.join(CONF.PATH.SCANNET_RAW2D, 'intrinsics.txt'))
        intrinsic_info['m_Width'], intrinsic_info['m_Height'] = 320, 240

    object2frame = image_3d_mapping(scan, image_list, color_list, img_names, pc_i, instances_i, extrinsic_list,
                                    intrinsic_info['m_intrinsic'], instance_names, intrinsic_info['m_Width'], intrinsic_info['m_Height'], 0, 0.20, scene_data)
    if object2frame:
        lock.acquire()
        with open(output_filepath, "wb") as f:
            pickle.dump(object2frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='test', choices=["train", "test", "validation"], help='train, test, validation')
    argparser.add_argument('--dataset', type=str, default='R3SCAN', choices=["R3SCAN", "SCANNET"], help='R3SCAN or SCANNET')
    argparser.add_argument('--parallel', action='store_true', help='parallel', required=False)

    args = argparser.parse_args()
    print("========= Deal with {} ========".format(args.mode))

    dataset, mode = args.dataset, args.mode
    export_path = CONF.PATH.R3SCAN if dataset == "R3SCAN" else CONF.PATH.SCANNET
    root = None

    if dataset == "SCANNET":
        root = os.path.join(CONF.PATH.SCANNET, "subgraphs")
    else:
        root = os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset")

    scene_data, selected_scans = read_json(root, mode)
    export_path = os.path.join(export_path, "views")

    scans = sorted(list(scene_data.keys()))
    print("Storing views in: ", export_path)

    skip_existing = True
    use_sam = False
    if args.parallel:
        process_map(partial(run, scene_data=scene_data, export_path=export_path, dataset=dataset), scans, max_workers=8, chunksize=4)
    else:
        for scan in tqdm(scans):
            run(scan, scene_data, export_path=export_path, dataset=dataset)
