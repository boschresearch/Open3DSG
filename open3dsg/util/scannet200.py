# This source code is from ScanNet
#   (https://github.com/ScanNet/ScanNet)
# Copyright (c) 2017 ScanNet authors
# This source code is licensed under the ScanNet license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

from open3dsg.util.scannet200_constants import *
from open3dsg.util.scannet200_splits import *
from open3dsg.util.utils_scannet import *
from open3dsg.config.define import SCANNET_LABELS_COMB
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load external constants

CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '.aggregation.json'
CLASS_IDs = VALID_CLASS_IDS_200


def nyu2scannet(scene_id, mesh_path, segments_file, aggregations_file, info_file):
    labels_pd = pd.read_csv(SCANNET_LABELS_COMB, sep='\t', header=0)

    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=' ')

    if 'axisAlignment' not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict['axisAlignment'].reshape(4, 4)

    pointcloud, faces_array = read_plymesh(mesh_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]
    alphas = pointcloud[:, -1]

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)
    pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(pointcloud, seg_indices, group, labels_pd, CLASS_IDs)

        labelled_pc[p_inds] = label_id
        instance_ids[p_inds] = group['id']

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)
    return labelled_pc, instance_ids