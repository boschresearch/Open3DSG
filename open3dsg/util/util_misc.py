# This source code is from 3DSSG
#   (https://github.com/ShunChengWu/3DSSG/tree/cvpr21)
# Copyright (c) 2021 3DSSG authors
# This source code is licensed under the BSD 2-Clause found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import json
import random
import numpy as np
from open3dsg.util import util_label
from open3dsg.util.util_label import scannet_label_ids, scannet_3rscan_label_mapping


def rand_24_bit():
    """Returns a random 24-bit integer"""
    return random.randrange(0, 16**6)


def color_dec():
    """Alias of rand_24 bit()"""
    return rand_24_bit()


def color_hex(num=rand_24_bit()):
    """Returns a 24-bit int in hex"""
    return "%06x" % num


def color_rgb(num=rand_24_bit()):
    """Returns three 8-bit numbers, one for each channel in RGB"""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])


def set_random_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_file_exist(path):
    if not os.path.exists(path):
        raise RuntimeError('Cannot open file. (', path, ')')


def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output


def read_relationships(read_file):
    relationships = []
    with open(read_file, 'r') as f:
        for line in f:
            relationship = line.rstrip().lower()
            relationships.append(relationship)
    return relationships


def load_semseg(json_file, name_mapping_dict=None, mapping=True):
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if labelName not in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if labelName not in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()  # segGroups["label"].lower()
    return instance2labelName


def load_semseg_scannet(pth_ply, pth_agg, pth_seg, label_names):
    from open3dsg.util import dataLoaderScanNet

    _, _, labels_gt, segments_gt = dataLoaderScanNet.load_scannet(pth_ply, pth_agg, pth_seg)
    instance2labelName = dict()
    size_segments_gt = dict()
    uni_seg_gt_ids = np.unique(segments_gt).tolist()
    for seg_id in uni_seg_gt_ids:
        indices = np.where(segments_gt == seg_id)
        seg = segments_gt[indices]
        labels = labels_gt[indices]
        uq_label = np.unique(labels).tolist()

        if len(uq_label) > 1:

            max_id = 0
            max_value = 0
            for id in uq_label:

                if len(labels[labels == id]) > max_value:
                    max_value = len(labels[labels == id])
                    max_id = id

            for label in uq_label:
                if label == max_id:
                    continue
                if len(labels[labels == id]) > 512:  # try to generate new segment
                    new_seg_idx = max(uni_seg_gt_ids)+1
                    uni_seg_gt_ids.append(new_seg_idx)
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = new_seg_idx
                else:
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = 0
                            labels_gt[idx] = 0  # set other label to 0
            seg = segments_gt[indices]
            labels = labels_gt[indices]
            uq_label = [max_id]

        if uq_label[0] == 0 or uq_label[0] > 40:
            name = 'none'
        else:
            name = util_label.NYU40_Label_Names[uq_label[0]-1]
            # print(name)

        if name not in label_names.values():
            name = 'none'

        # if label_type == 'ScanNet20':
        #     if name not in util_label.SCANNET20_Label_Names:
        #         name = 'none'

        size_segments_gt[seg_id] = len(seg)
        instance2labelName[seg_id] = name

    return instance2labelName


def load_semseg_scannet200(labels_gt, segments_gt):

    instance2labelName = dict()
    size_segments_gt = dict()
    uni_seg_gt_ids = np.unique(segments_gt).tolist()
    for seg_id in uni_seg_gt_ids:
        indices = np.where(segments_gt == seg_id)
        seg = segments_gt[indices]
        labels = labels_gt[indices]
        uq_label = np.unique(labels).tolist()
        indices = np.where(segments_gt == seg_id)
        seg = segments_gt[indices]
        labels = labels_gt[indices]
        uq_label = np.unique(labels).tolist()

        if len(uq_label) > 1:

            max_id = 0
            max_value = 0
            for id in uq_label:

                if len(labels[labels == id]) > max_value:
                    max_value = len(labels[labels == id])
                    max_id = id

            for label in uq_label:
                if label == max_id:
                    continue
                if len(labels[labels == id]) > 512:  # try to generate new segment
                    new_seg_idx = max(uni_seg_gt_ids)+1
                    uni_seg_gt_ids.append(new_seg_idx)
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = new_seg_idx
                else:
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = 0
                            labels_gt[idx] = 0  # set other label to 0
            seg = segments_gt[indices]
            labels = labels_gt[indices]
            uq_label = [max_id]

        if uq_label[0] == 0:
            name = 'none'
        else:
            scannet_name = scannet_label_ids[uq_label[0]]
            name = scannet_3rscan_label_mapping[scannet_name]

        size_segments_gt[seg_id] = len(seg)
        instance2labelName[seg_id] = name

    return instance2labelName
