# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import json
import argparse
import pickle
import multiprocessing
import trimesh
from scipy.spatial.distance import cdist
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from open3dsg.config.config import CONF

# from utils.fps.fps_utils import farthest_point_sampling

lock = multiprocessing.Lock()

pbar = None


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
    # m = np.max(pcl_, axis=0) - np.min(pcl_, axis=0)
    pcl_ = pcl_ / m
    pcl = pcl_
    return np.concatenate((pcl, rgbnrm_class), axis=1), centroid, m


def extract_axis_bbox(obj_pc):
    obb = {}  # centroid, axesLengths, normalizedAxes

    xyz_min = np.min(obj_pc, axis=0)
    xyz_max = np.max(obj_pc, axis=0)
    return np.concatenate((abs(xyz_max - xyz_min), np.mean(obj_pc, axis=0), np.array([0])), axis=0)


def farthest_point_sample(point, npoint):
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


def within_bbox(points, box):
    center = box['centroid']
    x, y, z = box['axesLengths']
    mask = (points[:, 0] >= center[0] - x/2) & (points[:, 0] <= center[0] + x/2) & \
        (points[:, 1] >= center[1] - y/2) & (points[:, 1] <= center[1] + y/2) & \
        (points[:, 2] >= center[2] - z/2) & (points[:, 2] <= center[2] + z/2)
    return mask


def within_bbox2(p, obb):
    p = p[:, :3]
    center = np.array(obb["centroid"])
    axis_len = np.array(obb["axesLengths"])
    axis_x = np.array(obb["normalizedAxes"][0:3])
    axis_y = np.array(obb["normalizedAxes"][3:6])
    axis_z = np.array(obb["normalizedAxes"][6:9])
    project_x = np.sum((axis_x[None]*(p - center)), axis=1)
    project_y = np.sum((axis_y[None]*(p - center)), axis=1)
    project_z = np.sum((axis_z[None]*(p - center)), axis=1)
    return (-axis_len[0]/2 <= project_x) & (project_x <= axis_len[0]/2) & (-axis_len[1]/2 <= project_y) & (project_y <= axis_len[1]/2) & (-axis_len[2]/2 <= project_z) & (project_z <= axis_len[2]/2)


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


class Preprocessor():
    def __init__(self, skip_existing=False, distance="avg"):
        self.skip_existing = skip_existing
        self.distance = distance
        self.word2idx = {}
        index = 0
        file = open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/classes.txt"), 'r')
        category = file.readline()[:-1]
        while category:
            self.word2idx[category] = index
            category = file.readline()[:-1]
            index += 1

        self.rel2idx = {}
        index = 0
        file = open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships.txt"), 'r')
        category = file.readline()[:-1]
        while category:
            self.rel2idx[category] = index
            category = file.readline()[:-1]
            index += 1

        self.boxes_train = json.load(open(CONF.PATH.R3SCAN_RAW+'/obj_boxes_train_refined.json', 'r'))
        self.boxes_val = json.load(open(CONF.PATH.R3SCAN_RAW+'/obj_boxes_val_refined.json', 'r'))

    def process_one_scan(self, relationships_scan, scan_id):

        # print(f"working on {relationships_scan['scan']}")
        # read point cloud from OBJ file
        scan = scan_id[:-2]
        pcl_array, rgb_array, normals_array = load_mesh(os.path.join(CONF.PATH.R3SCAN_RAW, "{}/mesh.refined.v2.obj".format(scan)),
                                                        os.path.join(CONF.PATH.R3SCAN_RAW, "{}/mesh.refined_0.png".format(scan)))

        object2fame = pickle.load(open(os.path.join(CONF.PATH.R3SCAN, 'views', f"{relationships_scan['scan']}_object2image.pkl"), "rb"))

        segments = {}  # key:segment id, value: points belong to this segment
        with open(os.path.join(CONF.PATH.R3SCAN_RAW, "{}/mesh.refined.0.010000.segs.v2.json".format(scan)), 'r') as f:
            seg_indices = json.load(f)["segIndices"]
            for index, i in enumerate(seg_indices):
                if i not in segments:
                    segments[i] = []
                if not index < pcl_array.shape[0]:
                    continue
                segments[i].append(np.concatenate((pcl_array[index], rgb_array[index], normals_array[index])))
                # segments[i].append(pcl_array[index])

        # group points of the same object
        # filter the object which does not belong to this split
        obj_id_list = []
        for k, _ in relationships_scan["objects"].items():
            obj_id_list.append(int(k))

        with open(os.path.join(CONF.PATH.R3SCAN_RAW, "{}/semseg.v2.json".format(scan)), 'r') as f:
            seg_groups = json.load(f)["segGroups"]
            objects = {}  # object mapping to its belonging points
            obb = {}  # obb in this scan split, size equals objects num
            labels = {}  # { id: 'category name', 6:'trash can'}
            seg2obj = {}  # mapping between segment and object id
            for o in seg_groups:
                id = o["id"]
                if id not in obj_id_list:  # no corresponding relationships in this split
                    continue
                if o["label"] not in self.word2idx:  # Categories not under consideration
                    continue
                labels[id] = o["label"]
                segs = o["segments"]
                objects[id] = []
                obb[id] = o["obb"]
                for i in segs:
                    seg2obj[i] = id
                    for j in segments[i]:
                        objects[id] = j.reshape(1, -1) if len(objects[id]) == 0 else np.concatenate((objects[id], j.reshape(1, -1)), axis=0)

        # sample and normalize point cloud
        obj_sample = 2000
        tight_bbox = []
        objects_center = []
        objects_scale = []
        remove_mini_objs = []
        objects_unnorm = {}
        for obj_id, obj_pcl in objects.items():
            try:
                test = True
                if not test:
                    bbox_params = self.boxes_train[scan][str(obj_id)] if scan in self.boxes_train.keys(
                    ) else self.boxes_val[scan][str(obj_id)]
                else:
                    bbox_params = {}
                bbox_params['axis_aligned'] = extract_axis_bbox(obj_pcl[:, :3])
                tight_bbox.append(bbox_params)
                pcl = farthest_point_sample(obj_pcl, obj_sample)
                objects_unnorm[obj_id] = pcl
                objects[obj_id], centroid, scale = pcl_normalize(pcl)
                objects_center.append(centroid)
                objects_scale.append(scale)
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
            if object2fame.get(str(o), None) is None:
                drop.append(o_i)
            else:
                object2frame_split[o] = object2fame[str(o)]
        if len(objects_id)-len(drop) < 4:
            print('too few visible objects, scene missalignment possible')
            return
            # raise Exception('too few visible objects, scene missalignment possible')
        for d in sorted(drop, reverse=True):
            objects_id.pop(d)
            objects_cat.pop(d)
            objects_pcl = np.delete(objects_pcl, d, 0)
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
            predicate_min_dist = []
            rels2frame_split = {}
            for rel in pairs:
                s, o = rel
                union_pcl = []
                pred_cls = np.zeros(len(self.rel2idx))
                for triple in triples:
                    if rel == triple[:2]:
                        pred_cls[triple[2]] = 1
                s, o = rel
                s_fids = [f[0] for f in object2fame[str(s)]]
                o_fids = [f[0] for f in object2fame[str(o)]]
                shared_frames = set(s_fids) & set(o_fids)
                s_o_frames = [(i, k, p, b, pix) for (i, k, p, b, pix) in object2fame[str(s)] if i in shared_frames]
                o_s_frames = [(i, k, p, b, pix) for (i, k, p, b, pix) in object2fame[str(o)] if i in shared_frames]
                # union_pcl_ = pcl_array[within_bbox2(pcl_array, obb[s]) | within_bbox2(pcl_array, obb[o])]
                # union_ins = inst_array[within_bbox2(pcl_array, obb[s]) | within_bbox2(pcl_array, obb[o])]
                # union_pcl_flag = np.zeros_like(union_ins)
                # union_pcl_flag[union_ins==s]=1
                # union_pcl_flag[union_ins==o]=2
                # union_pcl_ = np.concatenate((union_pcl_,union_pcl_flag),axis=1)
                tmp = (within_bbox2(pcl_array, obb[s]) | within_bbox2(pcl_array, obb[o]))
                rel2frame_split = [(s_f[0], s_f[1], o_f[1], s_f[2], o_f[2], s_f[3], o_f[3]) for s_f, o_f in zip(s_o_frames, o_s_frames)]
                for index, point in enumerate(pcl_array):
                    if seg_indices[index] not in seg2obj:
                        continue
                    if tmp[index]:  # judge_obb_intersect(point, obb[s]) or judge_obb_intersect(point, obb[o]):
                        if seg2obj[seg_indices[index]] == s:
                            point = np.append(point, 1)
                        elif seg2obj[seg_indices[index]] == o:
                            point = np.append(point, 2)
                        else:
                            point = np.append(point, 0)
                        union_pcl.append(point)
                union_point_cloud.append(union_pcl)
                predicate_cat.append(pred_cls.tolist())
                rels2frame_split[(s, o)] = rel2frame_split
            # sample and normalize point cloud
            rel_sample = 5000
            for index, _ in enumerate(union_point_cloud):
                pcl = np.array(union_point_cloud[index])
                s = pcl[pcl[:, -1] == 1]
                o = pcl[pcl[:, -1] == 2]
                c_s = np.mean(s[:, :3], axis=0)
                c_o = np.mean(o[:, :3], axis=0)
                dist = c_s-c_o
                try:
                    mag_dist = cdist(s[:, :3], o[:, :3])
                    min_idx = np.argmin(mag_dist)
                    point1_idx, point2_idx = np.unravel_index(min_idx, mag_dist.shape)
                    point1 = s[point1_idx, :3]
                    point2 = o[point2_idx, :3]
                    dist_min = point2 - point1
                except:
                    print(scan_id)
                    return

                pcl = farthest_point_sample(pcl, rel_sample)
                pcl_norm, _, _ = pcl_normalize(pcl)
                pcl_pad = np.zeros((rel_sample, 10))
                pcl_pad[:len(pcl)] = pcl_norm
                union_point_cloud[index] = pcl_pad
                predicate_dist.append(dist)
                predicate_min_dist.append(dist_min)
                predicate_num.append(len(pcl))
        except KeyError:
            print(scan_id)
            print(obb.keys())
            print(s, o, '\n')
            return

        # predicate_pcl_flag = []
        # for pcl in union_point_cloud:
        #     predicate_pcl_flag = pcl if len(predicate_pcl_flag) == 0 else np.concatenate((predicate_pcl_flag, pcl), axis=0)
        predicate_pcl_flag = np.stack(union_point_cloud, axis=0)

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
        data_dict["scan_id"] = scan_id
        data_dict["objects_id"] = objects_id  # object id
        data_dict["objects_cat"] = objects_cat  # object category
        data_dict["objects_num"] = objects_num
        data_dict["objects_pcl"] = objects_pcl.tolist()  # corresponding point cloud
        data_dict["objects_pcl_glob"] = objects_pcl_glob
        data_dict["objects_center"] = [l.tolist() for l in objects_center]
        data_dict["objects_scale"] = [l.tolist() for l in objects_scale]
        data_dict["predicate_cat"] = predicate_cat  # predicate id
        data_dict["predicate_num"] = predicate_num
        data_dict["predicate_pcl_flag"] = predicate_pcl_flag.tolist()  # corresponding point cloud in the union bounding box
        data_dict["predicate_dist"] = [l.tolist() for l in predicate_dist]
        data_dict["predicate_min_dist"] = [l.tolist() for l in predicate_min_dist]
        data_dict["pairs"] = pairs
        data_dict["edges"] = edges.tolist()
        data_dict["triples"] = triples
        data_dict["objects_count"] = len(objects_cat)
        data_dict["predicate_count"] = len(predicate_cat)
        data_dict["tight_bbox"] = tight_bbox
        data_dict["object2frame"] = object2frame_split
        data_dict["rel2frame"] = rels2frame_split
        data_dict["id2name"] = relationships_scan['objects']

        return data_dict

    def write_pickle(self, relationship):

        scan_id = relationship["scan"] + "-" + str(hex(relationship["split"]))[-1]
        folder = os.path.join(CONF.PATH.R3SCAN, "preprocessed", scan_id[:-2])
        filepath = os.path.join(folder, f"data_dict_{scan_id[-1]}.pkl")
        if os.path.exists(filepath) and self.skip_existing:
            # print('skipping already exists')
            return

        data_dict = self.process_one_scan(relationship, scan_id)
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

    relationships_train = json.load(open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships_train.json")))["scans"]
    relationships_val = json.load(open(os.path.join(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships_validation.json")))["scans"]

    relationships = relationships_train + relationships_val
    # relationships = relationships_val

    processor = Preprocessor(skip_existing=True, distance="min")

    import random
    random.shuffle(relationships)

    if args.parallel:
        process_map(processor.write_pickle, relationships, max_workers=16, chunksize=1)
    else:
        for r in tqdm(relationships):
            processor.write_pickle(r)
