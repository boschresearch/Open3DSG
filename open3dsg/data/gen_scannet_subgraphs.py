#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Some codes here are modified from 3DSSG https://github.com/ShunChengWu/3DSSG/blob/cvpr21/data_processing/gen_data_gt.py under BSD-2 License.

import argparse
import json
import math
import os
import random
from pathlib import Path
from enum import Enum
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import trimesh
import numpy as np
from open3dsg.config import define
from open3dsg.config.config import CONF
from open3dsg.util import util_label, util_misc
from open3dsg.util.util_search import SAMPLE_METHODS, find_neighbors
from open3dsg.util import dataLoaderScanNet
from open3dsg.util.scannet200 import nyu2scannet


def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument('--type', type=str, default='validation',
                        choices=['train', 'test', 'validation'], help="allow multiple rel pred outputs per pair", required=False)
    parser.add_argument('--relation', type=str, default='relationships', choices=['relationships_extended', 'relationships'])
    parser.add_argument('--target_scan', type=str, default='', help='')
    parser.add_argument('--label_type', type=str, default='ScanNet200', choices=['ScanNet20', 'ScanNet200'], help='label', required=False)

    # options
    parser.add_argument('--mapping', type=int, default=1,
                        help='map label from 3RScan to label_type. otherwise filter out labels outside label_type.')
    parser.add_argument('--v2', type=int, default=1, help='v2 version')
    parser.add_argument('--verbose', type=bool, default=False, help='verbal', required=False)
    parser.add_argument('--debug', action='store_true', help='debug', required=False)
    parser.add_argument('--parallel', action='store_true', help='parallel', required=False)

    # neighbor search parameters
    parser.add_argument('--search_method', type=str, choices=['BBOX', 'KNN'], default='BBOX', help='How to split the scene.')
    parser.add_argument('--radius_receptive', type=float, default=0.8, help='The receptive field of each seed.')

    # split parameters
    parser.add_argument('--split', type=int, default=1, help='Split scene into groups.')
    parser.add_argument('--radius_seed', type=float, default=1.3, help='The minimum distance between two seeds.')
    parser.add_argument('--min_segs', type=int, default=6, help='Minimum segments for each segGroup')
    parser.add_argument('--split_method', type=str, choices=['BBOX', 'KNN'], default='BBOX', help='How to split the scene.')

    return parser


def generate_groups(cloud: trimesh.points.PointCloud, segments: np.ndarray, distance: float = 1, bbox_distance: float = 0.75,
                    min_seg_per_group=5, segs_neighbors=None, instance2label=None):
    points = np.array(cloud.vertices.tolist())

    points = points[segments != 0]
    segments = segments[segments != 0]

    selected_indices = list()

    index = np.random.choice(range(len(points)), 1)
    selected_indices.append(index)
    should_continue = True
    while should_continue:
        distances_pre = None
        for index in selected_indices:
            point = points[index]
            # ignore z axis.
            distances = np.linalg.norm(points[:, 0:2]-point[:, 0:2], axis=1)
            if distances_pre is not None:
                distances = np.minimum(distances, distances_pre)
            distances_pre = distances
        selectable = np.where(distances > distance)[0]
        if len(selectable) < 1:
            should_continue = False
            break
        index = np.random.choice(selectable, 1)
        selected_indices.append(index)

    # Get segment groups#
    seg_group = list()

    #  Building Box Method #

    class SAMPLE_METHODS(Enum):
        BBOX = 1
        RADIUS = 2
    if args.split_method == 'BBOX':
        sample_method = SAMPLE_METHODS.BBOX
    elif args.split_method == 'KNN':
        sample_method = SAMPLE_METHODS.RADIUS

    if sample_method == SAMPLE_METHODS.BBOX:
        for index in selected_indices:
            point = points[index]
            min_box = (point-bbox_distance)[0]
            max_box = (point+bbox_distance)[0]

            filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])

            filtered_segments = segments[np.where(filter_mask > 0)[0]]
            segment_ids = np.unique(filtered_segments)
            # print('segGroup {} has {} segments.'.format(index,len(segment_ids)))
            if len(segment_ids) < min_seg_per_group:
                continue

            # problem -> 30/2 is still bigger than 10
            if len(segment_ids) > 9 and len(segment_ids) <= 18:
                if len(segment_ids)//2 > min_seg_per_group:
                    seg_group.append(segment_ids.tolist()[
                                     :(len(segment_ids)//2)])
                    seg_group.append(segment_ids.tolist()[
                                     (len(segment_ids)//2):])
                else:
                    seg_group.append(random.sample(segment_ids.tolist(), k=9))

            elif len(segment_ids) > 18:
                tmp = math.ceil(len(segment_ids) /
                                math.ceil((len(segment_ids)/9)))
                segment_ids_sub = [segment_ids[i:i + tmp].tolist()
                                   for i in range(0, len(segment_ids), tmp)]
                if len(segment_ids_sub[-1]) < min_seg_per_group:
                    if len(segment_ids_sub)-1 > 0:
                        free = 9-len(segment_ids_sub[0])
                        fits = [segment_ids_sub[-1][i:i + free]
                                for i in range(0, len(segment_ids_sub[-1]), free)][:len(segment_ids_sub)-1]
                        segment_ids_sub = [x + y for x,
                                           y in zip(segment_ids_sub[:-1], fits)]
                        seg_group.extend(segment_ids_sub)
                    else:
                        seg_group.extend(segment_ids_sub[:-1])

                else:
                    seg_group.extend(segment_ids_sub)

            else:
                seg_group.append(segment_ids.tolist())

    return seg_group


def process(scan_id, pth_3RScan, split_scene=True):

    pth_gt = os.path.join(pth_3RScan, scan_id, scan_id +
                          "_vh_clean_2.labels.ply")
    segseg_file_name = scan_id + \
        "_vh_clean_2.0.010000.segs.json"
    pth_semseg_file = os.path.join(pth_3RScan, scan_id, segseg_file_name)
    pth_ply = pth_gt
    pth_agg = os.path.join(pth_3RScan, scan_id, scan_id + ".aggregation.json")
    pth_seg = pth_semseg_file
    pth_info = os.path.join(pth_3RScan, scan_id, scan_id + ".txt")
    cloud_gt, points_gt, _, segments_gt = dataLoaderScanNet.load_scannet(pth_ply, pth_agg, pth_seg)
    labels_gt200, segments_gt200 = nyu2scannet(scan_id, pth_ply, pth_seg, pth_agg, pth_info)

    segs_neighbors = find_neighbors(points_gt, segments_gt, search_method, receptive_field=args.radius_receptive)
    relationships_new['neighbors'][scan_id] = segs_neighbors

    segment_ids = np.unique(segments_gt)
    segment_ids = segment_ids[segment_ids != 0]

    if args.label_type == "ScanNet20":
        _, label_name_mapping, _ = util_label.getLabelMapping(args.label_type)

        instance2labelName = util_misc.load_semseg_scannet(
            pth_ply, pth_agg, pth_seg, label_name_mapping)
    elif args.label_type == "ScanNet200":
        instance2labelName = util_misc.load_semseg_scannet200(
            labels_gt200, segments_gt200)
    else:
        raise RuntimeError("Label type not supported")

    if split_scene:
        seg_groups = generate_groups(cloud_gt, segments_gt, args.radius_seed, args.radius_receptive, args.min_segs,
                                     segs_neighbors=segs_neighbors, instance2label=instance2labelName)
        if args.verbose:
            print('final segGroups:', len(seg_groups))
    else:
        seg_groups = None

    #  Find and count all corresponding segments#
    size_segments_gt = dict()
    map_segment_pd_2_gt = dict()  # map segment_pd to segment_gt
    for segment_id in segment_ids:
        segment_indices = np.where(segments_gt == segment_id)[0]
        segment_points = points_gt[segment_indices]
        size_segments_gt[segment_id] = len(segment_points)
        map_segment_pd_2_gt[segment_id] = segment_id

    #  Save as ply #
    if debug:
        for seg, label_name in instance2labelName.items():
            segment_indices = np.where(segments_gt == seg)[0]
            if label_name != 'none':
                continue
            for index in segment_indices:
                cloud_gt.visual.vertex_colors[index][:3] = [0, 0, 0]
        cloud_gt.export(os.path.join(CONF.PATH.SCANNET, "subgraphs", 'tmp_gtcloud.ply'))

    # ' Save as relationship_*.json #
    list_relationships = list()
    if seg_groups is not None:
        for split_id, seg_group in enumerate(seg_groups):
            relationships = gen_relationship(scan_id, split_id, map_segment_pd_2_gt, instance2labelName, seg_group)
            if len(relationships["objects"]) == 0:
                continue
            list_relationships.append(relationships)

            #  check #
            for obj in relationships['objects']:
                assert (obj in seg_group)
            for rel in relationships['relationships']:
                assert (rel[0] in relationships['objects'])
                assert (rel[1] in relationships['objects'])
    else:
        relationships = gen_relationship(
            scan_id, 0, map_segment_pd_2_gt, instance2labelName)
        if len(relationships["objects"]) != 0 and len(relationships['relationships']) != 0:
            list_relationships.append(relationships)

    return scan_id, list_relationships, segs_neighbors, instance2labelName


def gen_relationship(scan_id: str, split: int, map_segment_pd_2_gt: dict, instance2labelName: dict,
                     target_segments: list = None) -> dict:
    # ' Save as relationship_*.json #
    relationships = dict()  # relationships_new["scans"].append(s)
    relationships["scan"] = scan_id
    relationships["split"] = split

    objects = dict()
    for seg, segment_gt in map_segment_pd_2_gt.items():
        if target_segments is not None:
            if seg not in target_segments:
                continue
        name = instance2labelName[segment_gt]
        if name == '-' or name == 'none':
            continue
        objects[int(seg)] = name  # labels_utils.NYU40_Label_Names[label-1]
    relationships["objects"] = objects

    split_relationships = list()

    relationships["relationships"] = split_relationships
    return relationships


if __name__ == '__main__':
    print("Generating ScanNet subgraphs")
    args = Parser().parse_args()
    debug = args.debug
    if args.search_method == 'BBOX':
        search_method = SAMPLE_METHODS.BBOX
    elif args.search_method == 'KNN':
        search_method = SAMPLE_METHODS.RADIUS

    input_dir = os.path.join(CONF.PATH.SCANNET_RAW, "scannet_3d", "data")
    assert os.path.exists(input_dir), f"{input_dir} does not exist"

    output_dir = os.path.join(CONF.PATH.SCANNET, "subgraphs")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching with {args.search_method}")

    #  Map label to 160#
    target_scan = []
    if args.target_scan != '':
        target_scan = util_misc.read_txt_to_list(args.target_scan)

    if args.type == 'train':
        scan_ids = util_misc.read_txt_to_list(define.SCANNET_SPLIT_TRAIN)
    elif args.type == 'validation':
        scan_ids = util_misc.read_txt_to_list(define.SCANNET_SPLIT_VAL)

    valid_scans = list()
    relationships_new = dict()
    relationships_new["scans"] = list()
    relationships_new['neighbors'] = dict()
    instance2label_scans = dict()
    counter = 0
    random.shuffle(scan_ids)
    print("Processing scans")
    if args.parallel:
        def process_with_args(scan_ids, scan_path, split_scene):
            partially_filled_proc = partial(process, pth_3RScan=scan_path, split_scene=split_scene)
            if debug:
                scan_ids = [scan_ids[0]]
            return process_map(partially_filled_proc, scan_ids, max_workers=8, chunksize=4)

        r = process_with_args(scan_ids, input_dir, split_scene=args.split)
        for scan_id, relationships, segs_neighbors, inst2labelName in r:
            valid_scans.append(scan_id)
            relationships_new["scans"] += relationships
            relationships_new['neighbors'][scan_id] = segs_neighbors
            instance2label_scans[scan_id] = inst2labelName
    else:
        for scan_id in tqdm(scan_ids):
            _, relationships, segs_neighbors, inst2labelName = process(scan_id, input_dir, split_scene=args.split)
            valid_scans.append(scan_id)
            relationships_new["scans"] += relationships
            relationships_new['neighbors'][scan_id] = segs_neighbors
            instance2label_scans[scan_id] = inst2labelName
            if debug:
                break

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pth_args = os.path.join(output_dir, 'args.json')
    print(f"Store used args at {pth_args}")
    with open(pth_args, 'w') as f:
        tmp = vars(args)
        json.dump(tmp, f, indent=2)

    pth_relationships_json = os.path.join(
        output_dir, "relationships_" + args.type + ".json")
    print(f"Store generated relationships at {pth_relationships_json}")
    with open(pth_relationships_json, 'w') as f:
        json.dump(relationships_new, f)

    pth_split = os.path.join(output_dir, args.type+'_scans.txt')
    print(f"Store used scans at {pth_split}")
    with open(pth_split, 'w') as f:
        for name in valid_scans:
            f.write(f'{name}\n')

    pth_inst2label = os.path.join(args.pth_out, 'instance2labels')
    print(f"Store used inst2label at {pth_split}")
    os.makedirs(pth_inst2label, exist_ok=True)
    for scan_id, inst2labelName in instance2label_scans.items():
        pth_inst2label_scan = os.path.join(pth_inst2label, scan_id+'_inst2label.json')
        with open(pth_inst2label_scan, 'w') as f:
            json.dump(inst2labelName, f)

    print("Done")
