# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""File containing the paths will be used in the code."""

import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.HOME = ""  # Your home directory
CONF.PATH.BASE = ""  # OpenSG directory
CONF.PATH.DATA = ""  # Root path for datasets

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# Original Datasets
CONF.PATH.R3SCAN_RAW = os.path.join(CONF.PATH.DATA, "3RScan")  # 3RScan original dataset directory
CONF.PATH.SCANNET_RAW = os.path.join(CONF.PATH.DATA, "SCANNET")  # ScanNet original dataset directory
CONF.PATH.SCANNET_RAW3D = os.path.join(CONF.PATH.SCANNET_RAW, "scannet_3d", "data")  # ScanNet original dataset directory
CONF.PATH.SCANNET_RAW2D = os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d")  # ScanNet original dataset directory

# Processed Dataset
# CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA, "OpenSG_3RScan")
# CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "OpenSG_ScanNet")
CONF.PATH.DATA_OUT = ""  # Output directory for processed datasets
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_3RScan")  # Output directory for processed 3RScan dataset
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA_OUT, "datasets", "OpenSG_ScanNet")  # Output directory for processed ScanNet dataset
CONF.PATH.CHECKPOINTS = os.path.join(CONF.PATH.DATA_OUT, "checkpoints")
CONF.PATH.FEATURES = os.path.join(CONF.PATH.DATA_OUT, "features")

# MLOps
CONF.PATH.MLOPS = os.path.join(CONF.PATH.BASE, "mlops")  # MLOps directory
CONF.PATH.MLFLOW = os.path.join(CONF.PATH.MLOPS, "opensg", "mlflow")  # Output directory for MLFlow data
CONF.PATH.TENSORBOARD = os.path.join(CONF.PATH.MLOPS, "opensg", "tensorboards")  # Output directory for Tensorboard data

for _, path in CONF.PATH.items():
    assert os.path.exists(path), f"{path} does not exist"