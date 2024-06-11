# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os

from open3dsg.config.config import CONF

ROOT_PATH = CONF.PATH.R3SCAN_RAW
SCANNET_DATA_PATH = '/store/datasets/scannet/scans'
SCANNET_SPLIT_TRAIN = os.path.join(CONF.PATH.SCANNET_RAW, "scannetv2_train.txt")
SCANNET_SPLIT_VAL = os.path.join(CONF.PATH.SCANNET_RAW, "scannetv2_val.txt")
SCANNET_LABELS_COMB = os.path.join(CONF.PATH.SCANNET_RAW, "scannetv2-labels.combined.tsv")

FILE_PATH = ROOT_PATH
Scan3RJson_PATH = FILE_PATH+'3RScan.json'
LABEL_MAPPING_FILE = FILE_PATH+'3RScan.v2 Semantic Classes - Mapping.csv'
CLASS160_FILE = FILE_PATH+'classes160.txt'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'


NAME_SAME_PART = 'same part'
