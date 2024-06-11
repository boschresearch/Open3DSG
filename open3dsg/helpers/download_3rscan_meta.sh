#!/bin/bash
# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

wget http://campar.in.tum.de/files/3RScan/rescans.txt
wget http://campar.in.tum.de/files/3RScan/train_scans.txt
wget http://campar.in.tum.de/files/3RScan/val_scans.txt
wget http://campar.in.tum.de/files/3RScan/3RScan.json

pip install gdown
gdown --folder https://drive.google.com/drive/folders/1onEfhRDZWFvVsrwcEa3jTr3m-otYTFty
gdown --folder https://drive.google.com/drive/folders/1purK2F0cFrEDwpPNYQVPHX_LJyL1reur