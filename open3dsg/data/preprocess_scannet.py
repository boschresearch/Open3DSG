# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# ATTENTION HERE: since the dataset has some some problem, i have made some modification in the dataset,
# So if want to transfer the code to another machine, it is necessary to transfer the data meanwhile.
import os
import argparse
import json
import multiprocessing
import pickle
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from open3dsg.config.config import CONF
from open3dsg.config.define import SCANNET_LABELS_COMB
from open3dsg.util.scannet200_constants import CLASS_LABELS_200
import pandas as pd

lock = multiprocessing.Lock()

pbar = None
labels_pd = pd.read_csv(SCANNET_LABELS_COMB, sep='\t', header=0)


def scannet_get_instance_ply(plydata, segs, aggre):
    #  map idx to segments
    seg_map = dict()
    for idx in range(len(segs['segIndices'])):
        seg = segs['segIndices'][idx]
        if seg in seg_map:
            seg_map[seg].append(idx)
        else:
            seg_map[seg] = [idx]

    #  Group segments
    aggre_seg_map = dict()
    for segGroup in aggre['segGroups']:
        aggre_seg_map[segGroup['id']] = list()
        for seg in segGroup['segments']:
            aggre_seg_map[segGroup['id']].extend(seg_map[seg])
    assert (len(aggre_seg_map) == len(aggre['segGroups']))
    # print('num of aggre_seg_map:',len(aggre_seg_map))

    #  Over write label to segments
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
    #  Load GT
    plydata = trimesh.load(pth_ply, process=False)
    num_verts = plydata.vertices.shape[0]
    if verbose:
        print('num of verts:', num_verts)

    #  Load segment file
    with open(pth_seg) as f:
        segs = json.load(f)
    if verbose:
        print('len(aggre[\'segIndices\']):', len(segs['segIndices']))
    segment_ids = list(np.unique(np.array(segs['segIndices'])))  # get unique segment ids
    if verbose:
        print('num of unique ids:', len(segment_ids))

    #  Load aggregation file
    with open(pth_agg) as f:
        aggre = json.load(f)

    plydata, instances = scannet_get_instance_ply(plydata, segs, aggre)
    labels = plydata.metadata['_ply_raw']['vertex']['data']['label'].flatten()

    return plydata, plydata.vertices, labels, instances


def extract_bbox(obj_pc):
    obb = {}  # centroid, axesLengths, normalizedAxes
    obb['centroid'] = np.mean(obj_pc, axis=0)
    obb['normalizedAxes'] = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])
    xyz_min = np.min(obj_pc, axis=0)
    xyz_max = np.max(obj_pc, axis=0)
    obb['axesLengths'] = abs(xyz_max - xyz_min)
    return obb


def extract_axis_bbox(obj_pc):
    obb = {}  # centroid, axesLengths, normalizedAxes

    xyz_min = np.min(obj_pc, axis=0)
    xyz_max = np.max(obj_pc, axis=0)
    return np.concatenate((abs(xyz_max - xyz_min), np.mean(obj_pc, axis=0), np.array([0])), axis=0)


def load_mesh(mesh_path, texture_path):
    mesh = trimesh.load(mesh_path, process=False)
    im = Image.open(texture_path)
    tex = trimesh.visual.TextureVisuals(image=im)
    mesh.visual.texture = tex

    return mesh.vertices, mesh.visual.to_color().vertex_colors[:, :3], mesh.vertex_normals


def pcl_normalize(pcl):
    rgbnrm_class = pcl[:, 3:]
    pcl_ = pcl[:, :3]
    centroid = np.mean(pcl_, axis=0)
    pcl_ = pcl_ - centroid
    m = np.max(np.sqrt(np.sum(pcl_ ** 2, axis=1)))
    pcl_ = pcl_ / m
    pcl = pcl_
    return np.concatenate((pcl, rgbnrm_class), axis=1), centroid


def farthest_point_sample(point, npoint):
    '''
    https://github.com/salesforce/ULIP/blob/2fed30f12501b0f947ee9e84b5d8620d04545163/models/pointnet2/pointnet2_utils.py#L63
    '''
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        return point
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def judge_obb_intersect(p, obb):
    # judge one point is or not in the obb
    p = p[:3]
    center = np.array(obb["centroid"])
    axis_len = np.array(obb["axesLengths"])
    axis_x = np.array(obb["normalizedAxes"][0:3])
    axis_y = np.array(obb["normalizedAxes"][3:6])
    axis_z = np.array(obb["normalizedAxes"][6:9])
    project_x = axis_x.dot(p - center)
    project_y = axis_y.dot(p - center)
    project_z = axis_z.dot(p - center)
    return -axis_len[0]/2 <= project_x <= axis_len[0]/2 and \
           -axis_len[1]/2 <= project_y <= axis_len[1]/2 and \
           -axis_len[2]/2 <= project_z <= axis_len[2]/2


def within_bbox(points, box):
    center = box['centroid']
    x, y, z = box['axesLengths']
    mask = (points[:, 0] >= center[0] - x/2) & (points[:, 0] <= center[0] + x/2) & \
        (points[:, 1] >= center[1] - y/2) & (points[:, 1] <= center[1] + y/2) & \
        (points[:, 2] >= center[2] - z/2) & (points[:, 2] <= center[2] + z/2)
    return mask


class Preprocessor():
    def __init__(self, skip_existing=False):
        self.skip_existing = skip_existing
        self.word2idx = {}
        index = 0

        for category in CLASS_LABELS_200:
            self.word2idx[category] = index
            index += 1

        self.rel2idx = {}
        index = 0
        file = open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset", "relationships.txt"), 'r')
        category = file.readline().rstrip()
        while category:
            self.rel2idx[category] = index
            category = file.readline().rstrip()
            index += 1

    def process_one_scan(self, relationships_scan, scan_id):

        # print(f"working on {relationships_scan['scan']}")
        scan_id, split = scan_id.split('-')
        pth_scannet = CONF.PATH.SCANNET_RAW3D
        pth_ply = os.path.join(pth_scannet, scan_id, scan_id + "_vh_clean_2.labels.ply")
        pth_cld = os.path.join(pth_scannet, scan_id, scan_id + "_vh_clean_2.ply")
        segseg_file_name = scan_id + "_vh_clean_2.0.010000.segs.json"
        pth_seg = os.path.join(pth_scannet, scan_id, segseg_file_name)
        pth_agg = os.path.join(pth_scannet, scan_id, scan_id + "_vh_clean.aggregation.json")
        pth_info = os.path.join(pth_scannet, scan_id, scan_id + ".txt")

        pcl = trimesh.load(pth_cld, process=False)
        pcl_array, rgb_array, normals_array = np.asarray(pcl.vertices), np.asarray(
            pcl.visual.vertex_colors[:, :3]), np.asarray(pcl.vertex_normals)
        cloud_gt, points_gt, labels_gt, segments_gt = load_scannet(pth_ply, pth_seg, pth_agg)

        object2fame = pickle.load(open(os.path.join(CONF.PATH.SCANNET, 'views',
                                  f"{relationships_scan['scan']}_object2image.pkl"), "rb"))

        with open(pth_seg) as f:
            segments = json.load(f)
            seg_indices = np.array(segments['segIndices'])

        # Load Aggregations file
        with open(pth_agg) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation['segGroups'])

        # group points in the same segment
        segments = {}  # key:segment id, value: points belong to this segment
        for index, i in enumerate(seg_indices):
            if i not in segments:
                segments[i] = []
            if not index < pcl_array.shape[0]:
                continue
            segments[i].append(np.concatenate((pcl_array[index], rgb_array[index], normals_array[index])))

        # group points of the same object
        # filter the object which does not belong to this split
        obj_id_list = []
        for k, _ in relationships_scan["objects"].items():
            obj_id_list.append(int(k))
        if len(obj_id_list) > 10 or len(obj_id_list) < 4:
            return None

        objects = {}  # object mapping to its belonging points
        obb = {}  # obb in this scan split, size equals objects num
        labels = {}  # { id: 'category name', 6:'trash can'}
        seg2obj = {}  # mapping between segment and object id
        for o in seg_groups:
            id = o["id"]
            if id not in obj_id_list:  # no corresponding relationships in this split
                continue
            # if labels_pd[labels_pd['raw_category'] == o['label']]['category'].iloc[0]  not in self.word2idx:  # Categories not under consideration
            #     continue
            labels[id] = labels_pd[labels_pd['raw_category'] == o['label']]['category'].iloc[0]  # o["label"]
            segs = o["segments"]
            objects[id] = []

            for i in segs:
                seg2obj[i] = id
                for j in segments[i]:
                    objects[id] = j.reshape(1, -1) if len(objects[id]) == 0 else np.concatenate((objects[id], j.reshape(1, -1)), axis=0)
            obb[id] = extract_bbox(objects[id][:, :3])

        # sample and normalize point cloud
        obj_sample = 1000
        tight_bbox = []
        objects_center = []
        remove_mini_objs = []
        objects_unnorm = {}
        for obj_id, obj_pcl in objects.items():
            try:
                tight_bbox.append(extract_axis_bbox(obj_pcl[:, :3]))
                # tight_bbox.append(self.boxes_train[scan][str(obj_id)] if scan in self.boxes_train.keys() else self.boxes_val[scan][str(obj_id)])
                pcl = farthest_point_sample(obj_pcl, obj_sample)
                objects_unnorm[obj_id] = pcl
                objects[obj_id], centroid = pcl_normalize(pcl)
                objects_center.append(centroid)
            except KeyError:
                remove_mini_objs.append(obj_id)
                continue

        for id in remove_mini_objs:
            objects.pop(id)

        objects_id = []
        objects_cat = []
        objects_pcl = []
        objects_num = []
        objects_pcl_glob = []
        for k, v in objects.items():
            objects_id.append(k)
            objects_cat.append(self.word2idx[labels[k]])
            objects_num = objects_num + [len(v)]
            object_pts = np.zeros((obj_sample, 9))
            object_pts[:len(v)] = v
            objects_pcl.append(object_pts)
            objects_pcl_unnorm = np.zeros((obj_sample, 9))
            objects_pcl_unnorm[:len(objects_unnorm[k])] = objects_unnorm[k]
            objects_pcl_glob.append(objects_pcl_unnorm)
        objects_pcl = np.stack(objects_pcl, axis=0)
        objects_pcl_glob = np.stack(objects_pcl_glob, axis=0)

        # (frame, pixels, vis_fraction, bbox)
        object2frame_split = {}
        drop = []
        for o_i, o in enumerate(objects_id):
            if object2fame.get(o, None) is None:
                drop.append(o_i)
            else:
                object2frame_split[o] = object2fame[o]

        if len(objects_id)-len(drop) < 4:
            raise Exception('too few visible objects, scene missalignment possible')
        for d in sorted(drop, reverse=True):
            objects_id.pop(d)
            objects_cat.pop(d)
            objects_pcl = np.delete(objects_pcl, d, 0)
            objects_pcl_glob = np.delete(objects_pcl_glob, d, 0)
            objects_num.pop(d)

        # predicate input of PointNet, including points in the union bounding box of subject and object
        # here consider every possible combination between objects, if there doesn't exist relation in the training file,
        # add the relation with the predicate id replaced by 0
        triples = []
        pairs = []
        relationships_triples = relationships_scan["relationships"]
        for triple in relationships_triples:
            if (triple[0] not in objects_id) or (triple[1] not in objects_id) or (triple[0] == triple[1]):
                continue
            triples.append(triple[:3])
            if triple[:2] not in pairs:
                pairs.append(triple[:2])
        for i in objects_id:
            for j in objects_id:
                if i == j or [i, j] in pairs:
                    continue
                triples.append([i, j, 0])   # supplement the 'none' relation
                pairs.append(([i, j]))

        s = 0
        o = 0
        pcl_array = np.concatenate((pcl_array, rgb_array, normals_array), axis=1)
        try:
            union_point_cloud = []
            predicate_cat = []
            predicate_num = []
            predicate_dist = []
            rels2frame_split = {}
            for rel in pairs:

                s, o = rel
                s_fids = [f[0] for f in object2fame[s]]
                o_fids = [f[0] for f in object2fame[o]]
                shared_frames = set(s_fids) & set(o_fids)
                s_o_frames = [(i, k, p, b, pix) for (i, k, p, b, pix) in object2fame[s] if i in shared_frames]
                o_s_frames = [(i, k, p, b, pix) for (i, k, p, b, pix) in object2fame[o] if i in shared_frames]
                # (frame, s_pixels, o_pixels, s_vis, o_vis, s_bbox, o_bbox)
                rel2frame_split = [(s_f[0], s_f[1], o_f[1], s_f[2], o_f[2], s_f[3], o_f[3]) for s_f, o_f in zip(s_o_frames, o_s_frames)]
                union_pcl = []
                pred_cls = np.zeros(len(self.rel2idx))
                for triple in triples:
                    if rel == triple[:2]:
                        pred_cls[triple[2]] = 1
                union_pcl = pcl_array[within_bbox(pcl_array, obb[s]) | within_bbox(pcl_array, obb[o])]
                union_ins = segments_gt[within_bbox(pcl_array, obb[s]) | within_bbox(pcl_array, obb[o])]
                union_pcl_flag = np.zeros_like(union_ins)
                union_pcl_flag[union_ins == s] = 1
                union_pcl_flag[union_ins == o] = 2
                union_pcl = np.concatenate((union_pcl, union_pcl_flag[:, None]), axis=1)

                union_point_cloud.append(union_pcl)
                predicate_cat.append(pred_cls)
                rels2frame_split[(s, o)] = rel2frame_split
            # sample and normalize point cloud
            rel_sample = 3000
            for index, _ in enumerate(union_point_cloud):
                pcl = np.array(union_point_cloud[index])
                s = pcl[pcl[:, -1] == 1]
                o = pcl[pcl[:, -1] == 2]
                c_s = np.mean(s[:, :3], axis=0)
                c_o = np.mean(o[:, :3], axis=0)
                dist = c_s-c_o
                pcl = farthest_point_sample(pcl, rel_sample)
                predicate_num.append(len(pcl))
                pcl_norm, _ = pcl_normalize(pcl)
                pcl_pad = np.zeros((rel_sample, 10))
                pcl_pad[:len(pcl)] = pcl_norm
                union_point_cloud[index] = pcl_pad
                predicate_dist.append(dist)

        except KeyError:
            print(scan_id)
            print(obb.keys())
            print(s, o, '\n')
            return

        predicate_pcl_flag = np.stack(union_point_cloud, axis=0)
        # for pcl in union_point_cloud:
        #     predicate_pcl_flag = pcl if len(predicate_pcl_flag) == 0 else np.concatenate((predicate_pcl_flag, pcl), axis=0)

        object_id2idx = {}  # convert object id to the index in the tensor
        for index, v in enumerate(objects_id):
            object_id2idx[v] = index
        s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
        s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

        for index, v in enumerate(s):
            s[index] = object_id2idx[v]  # s_idx
        for index, v in enumerate(o):
            o[index] = object_id2idx[v]  # o_idx
        edges = np.stack((s, o), axis=1)    # edges is used for the input of the GCN module

        # # since point cloud in 3DSGG has been processed, there is no need to sample any more => actually need
        # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)

        data_dict = {}
        data_dict["scan_id"] = scan_id+'-'+split
        data_dict["objects_id"] = objects_id  # object id
        data_dict["objects_cat"] = objects_cat  # object category
        data_dict["objects_num"] = objects_num
        data_dict["objects_pcl"] = objects_pcl  # corresponding point cloud
        data_dict["objects_pcl_glob"] = objects_pcl_glob  # corresponding point cloud
        data_dict["objects_center"] = objects_center
        data_dict["predicate_cat"] = predicate_cat  # predicate id
        data_dict["predicate_num"] = predicate_num
        data_dict["predicate_pcl_flag"] = predicate_pcl_flag  # corresponding point cloud in the union bounding box
        data_dict["predicate_dist"] = predicate_dist
        data_dict["pairs"] = pairs
        data_dict["edges"] = edges
        data_dict["triples"] = triples
        data_dict["objects_count"] = len(objects_cat)
        data_dict["predicate_count"] = len(predicate_cat)
        data_dict["tight_bbox"] = tight_bbox
        data_dict["scannet"] = True
        data_dict["object2frame"] = object2frame_split
        data_dict["rel2frame"] = rels2frame_split
        data_dict["id2name"] = relationships_scan['objects']

        return data_dict

    def write_pickle(self, relationship):
        # redo = {'scene0358_01':2}
        # if not (relationship['scan'] in redo.keys()):# and relationship['split'] == redo[relationship['scan']]):
        #     return

        scan_id = relationship["scan"] + "-" + str(hex(relationship["split"]))[-1]
        folder = os.path.join(CONF.PATH.SCANNET, "preprocessed", scan_id.split('-')[0])
        filepath = os.path.join(folder, f"data_dict_{scan_id.split('-')[1]}.pkl")
        if os.path.exists(filepath) and self.skip_existing:
            # print('skipping already exists')
            return

        try:
            data_dict = self.process_one_scan(relationship, scan_id)
        except Exception as e:
            print('something went wrong, ', relationship)
            print(e)
            return
        if data_dict is None:
            return
        # process needs lock to write into disk
        lock.acquire()
        scan_id = data_dict["scan_id"]
        os.makedirs(folder, exist_ok=True)

        # print("{}/data_dict_{}.pkl".format(scan_id[:-2], scan_id[-1]))
        # with open(path, 'w') as f:
        #     f.write(json.dumps(data_dict, indent=4))
        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        lock.release()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--parallel', action='store_true', help='parallel', required=False)
    args = argparser.parse_args()
    relationships_train = json.load(open(os.path.join(CONF.PATH.SCANNET, "subgraphs", "relationships_train.json")))["scans"]
    relationships_val = json.load(open(os.path.join(CONF.PATH.SCANNET, "subgraphs", "relationships_validation.json")))["scans"]

    relationships = relationships_train + relationships_val
    processor = Preprocessor(skip_existing=True)

    import random
    random.shuffle(relationships)

    if args.parallel:
        process_map(processor.write_pickle, relationships, max_workers=16, chunksize=1)
    else:
        for r in tqdm(relationships):
            processor.write_pickle(r)
