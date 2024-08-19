#!/bin/bash
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

wget https://campar.in.tum.de/public_datasets/3RScan/rescans.txt
wget https://campar.in.tum.de/public_datasets/3RScan/train_scans.txt
wget https://campar.in.tum.de/public_datasets/3RScan/val_scans.txt
wget https://campar.in.tum.de/public_datasets/3RScan/3RScan.json

pip install gdown
gdown --folder https://drive.google.com/drive/folders/1onEfhRDZWFvVsrwcEa3jTr3m-otYTFty
gdown --folder https://drive.google.com/drive/folders/1purK2F0cFrEDwpPNYQVPHX_LJyL1reur