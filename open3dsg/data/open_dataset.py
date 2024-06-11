# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import os.path as osp
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from tqdm.contrib.concurrent import process_map
from functools import partial
from multiprocessing import Manager

from itertools import accumulate
import random
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torch import nn
import torchvision.transforms.functional as F

from open3dsg.config.config import CONF


class ResizeMaxSize(nn.Module):
    '''
    https://github.com/mlfoundations/open_clip/blob/f692ec95e1bf30d50aeabe2fd32008cdff53ef5e/src/open_clip/transform.py#L26
    '''
    def __init__(self, max_size, interpolation=BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        elif height != width:
            pad_h = self.max_size - height
            pad_w = self.max_size - width
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)

        return img


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        ResizeMaxSize(n_px, interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def scale_bbox(bbox, scale, img):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    new_width = width * scale
    new_height = height * scale

    return (
        max(0, x_center - new_width / 2),
        max(0, y_center - new_height / 2),
        min(img.size[0], x_center + new_width / 2),
        min(img.size[1], y_center + new_height / 2)
    )


def enclosing_bbox(bbox1, bbox2):
    return min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])


def read_class(path):
    file = open(os.path.join(CONF.PATH.R3SCAN_RAW, path), 'r')
    category = file.readline().rstrip()
    word_dict = []
    while category:
        word_dict.append(category)
        category = file.readline().rstrip()
    return word_dict


clip_scales = [1.0, 1.5, 1.2, 2.0, 2.5, 1.8, 2.2, 2.8]  # possible clip scales order by diversty


class DataDict:
    def __init__(self, data_dict):
        self.scan_id = data_dict["scan_id"]
        self.objects_id = np.array(data_dict["objects_id"]).astype(np.uint8)
        self.objects_cat = np.array(data_dict["objects_cat"]).astype(np.uint8)
        self.objects_num = np.array(data_dict["objects_num"]).astype(np.int16)
        self.objects_pcl = np.array(data_dict["objects_pcl"]).astype(np.float32)
        self.objects_center = np.array(data_dict["objects_center"]).astype(np.float32)
        self.objects_scale = np.array(data_dict["objects_scale"]).astype(np.float32)
        self.predicate_cat = np.array(data_dict["predicate_cat"]).astype(np.uint8)
        self.predicate_num = np.array(data_dict["predicate_num"]).astype(np.int16)
        self.predicate_pcl_flag = np.array(data_dict["predicate_pcl_flag"]).astype(np.float32)
        self.predicate_dist = np.array(data_dict["predicate_dist"]).astype(np.float32)
        if "predicate_min_dist" in data_dict.keys():
            self.predicate_min_dist = data_dict['predicate_min_dist']
        self.pairs = np.array(data_dict["pairs"]).astype(np.uint8)
        self.edges = np.array(data_dict["edges"]).astype(np.uint8)
        self.triples = np.array(data_dict["triples"]).astype(np.uint8)
        self.objects_count = np.array(data_dict["objects_count"]).astype(np.int16)
        self.predicate_count = np.array(data_dict["predicate_count"]).astype(np.int16)
        self.tight_bbox = data_dict["tight_bbox"]
        self.dataset = data_dict["dataset"]
        self.obj2frame = data_dict["object2frame"]
        self.rel2frame = data_dict["rel2frame"]
        self.rel2frame_mask = data_dict['rel2frame_mask']
        self.scene_id = data_dict["scene_id"]
        self.id2name = data_dict['id2name']


def _load_data_tqdm(shared_list, relationship):
    """
    Load all data into ram for faster training
    """
    path = os.path.join(CONF.PATH.R3SCAN, "preprocessed",
                        "{}/data_dict_{}.pkl".format(relationship["scan"], str(hex(relationship["split"]))[-1]))
    try:
        data_dict = pickle.load(open(path, "rb"))
        mask = {key: bool(value) for key, value in data_dict['rel2frame'].items()}
        data_dict['rel2frame_mask'] = mask
        data_dict["dataset"] = "3rscan"
        if "predicate_min_dist" not in data_dict.keys():
            data_dict['predicate_min_dist'] = data_dict['predicate_dist']
        data_dict["scene_id"] = relationship["scan"]
        shared_list.append(DataDict(data_dict))
    except Exception as e:
        print(e, path)


def _load_data_scannet_tqdm(shared_list, relationship):
    """
    Load all data into ram for faster training
    """
    path = os.path.join(CONF.PATH.SCANNET, "preprocessed",
                        "{}/data_dict_{}.pkl".format(relationship["scan"], str(hex(relationship["split"]))[-1]))
    try:
        data_dict = pickle.load(open(path, "rb"))
        mask = {key: bool(value) for key, value in data_dict['rel2frame'].items()}
        data_dict['rel2frame_mask'] = mask
        data_dict["dataset"] = "scannet"
        data_dict["objects_scale"] = [-1]*len(data_dict['objects_cat'])
        data_dict["scene_id"] = relationship["scan"]

        shared_list.append(DataDict(data_dict))
    except Exception as e:
        print(e, path)


class Open2D3DSGDataset(Dataset):

    def __init__(self, relationships_R3SCAN=None,
                 relationships_scannet=None,
                 img_dim=224,  # self.img_dim for 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 336 for ViT-L/14@336px
                 rel_img_dim=None,
                 mini=False,
                 openseg=False,
                 top_k_frames=5,
                 scales=3,
                 max_objects=9,
                 max_rels=72,
                 load_features=None,
                 blip=False,
                 llava=False,
                 half=False
                 ):
        self.img_dim = img_dim
        self.rel_img_dim = rel_img_dim if rel_img_dim else img_dim
        self.preprocess_obj = _transform(self.img_dim)
        self.preprocess_rel = _transform(self.rel_img_dim)
        self.openseg = openseg
        self.relationships_R3SCAN = relationships_R3SCAN
        self.relationships_scannet = relationships_scannet
        self.load_features = load_features
        self.blip = blip
        self.llava = llava
        self.max_objs = max_objects
        self.max_rels = max_rels

        # these are hard-coded at the moment
        self.obj_vis_crit = 0.3
        self.obj_mask_crit = 0.2
        self.rel_vis_crit = 0.3

        if mini:
            if self.relationships_R3SCAN:
                self.relationships_R3SCAN = self.relationships_R3SCAN[:10]
            if self.relationships_scannet:
                self.relationships_scannet = self.relationships_scannet[:10]
        if half:
            if self.relationships_R3SCAN:
                self.relationships_R3SCAN = self.relationships_R3SCAN[:200]
            if self.relationships_scannet:
                self.relationships_scannet = self.relationships_scannet[:200]

        self.top_k_frames = top_k_frames
        self.scales = scales

        self.scene_data = []
        manager = Manager()
        shared_list = manager.list()

        if self.relationships_R3SCAN:
            self.obj_vis_crit -= 0.1  # r3scan images are smaller than scannet lets adjust
            self.obj_mask_crit -= 0.1
            self.rel_vis_crit -= 0.1
            # Load all data into ram for faster training
            process_map(partial(_load_data_tqdm, shared_list), self.relationships_R3SCAN, max_workers=8, chunksize=1)

        if self.relationships_scannet:
            # Load all data into ram for faster training
            process_map(partial(_load_data_scannet_tqdm, shared_list), self.relationships_scannet, max_workers=8, chunksize=1)

        self.scene_data = shared_list
        self.pixel_data = {}

    def __len__(self):
        return len(self.scene_data)

    def obj_frame_selection(self, obj2frame, scene_id, dataset, top_k=3, scales=2):
        reference_imgs = []

        obj2frame_mask = {k: top_k*scales for k, v in obj2frame.items()}
        obj2frame_lst = list(obj2frame.values())
        r3scan_bias = 0.5 if dataset == '3rscan' else 1
        for i, (k, v) in enumerate(obj2frame.items()):
            # (frame, pixels, vis_fraction, bbox)
            obj = v  # obj2frame_lst[i]
            frames, pixels, vis, bbox, pixel_ids = tuple(zip(*obj))
            frames, pixels, vis, bbox = np.array(frames), np.array(pixels), np.array(vis), np.array(bbox)

            # filter all where more than 10% of the object is visible
            vis_criterion = ((vis > self.obj_vis_crit) & (
                (bbox[:, 2]-bbox[:, 0])*(bbox[:, 3]-bbox[:, 1]) > 500*r3scan_bias)) | ((bbox[:, 2]-bbox[:, 0])*(bbox[:, 3]-bbox[:, 1]) > 8000*r3scan_bias)
            if vis_criterion.sum() > 1:
                frames = frames[vis_criterion]
                pixels = pixels[vis_criterion]
                bbox = bbox[vis_criterion]
                vis = vis[vis_criterion]
            else:
                blanks = torch.zeros(((scales)*top_k, 3, self.img_dim, self.img_dim))
                reference_imgs.append(blanks)
                obj2frame_mask[list(obj2frame.keys())[i]] = 0
                continue

            ids = range(len(vis))
            vis, ids = zip(*sorted(zip(vis, ids), reverse=True))
            frames = frames[np.array(ids)]
            bbox = bbox[np.array(ids)]
            vis, frames, bbox = vis[:top_k], frames[:top_k], bbox[:top_k]
            selected = list(zip(vis, frames, bbox))

            if dataset == 'scannet':
                imgs = [Image.open(os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d", scene_id, "color", s[1])) for s in selected]
            else:
                imgs = [Image.open(os.path.join(CONF.PATH.R3SCAN_RAW, scene_id, 'sequence', s[1])).resize((224, 172)) for s in selected]

            cropped_imgs = [img.crop(scale_bbox(s[2], sc, img)) for img, s in zip(imgs, selected) for sc in clip_scales[:scales]]
            if dataset == '3rscan':
                cropped_imgs = [img.rotate(-90, expand=True) for img in cropped_imgs]

            if not np.array([((img.size[0] > 0) & (img.size[1] > 0)) for img in cropped_imgs]).all():
                print('zero area')
            processed_imgs = torch.stack([self.preprocess_obj(im) for im in cropped_imgs], dim=0)

            if len(cropped_imgs) < (scales)*top_k:
                blanks = torch.zeros(((scales)*top_k-len(cropped_imgs), 3, self.img_dim, self.img_dim))
                processed_imgs = torch.cat((processed_imgs, blanks), dim=0)
                obj2frame_mask[list(obj2frame.keys())[i]] = len(cropped_imgs)

            reference_imgs.append(processed_imgs)

        return reference_imgs, obj2frame_mask

    def obj_pixel_selection(self, obj2frame, scene_id, dataset, top_k=3):
        reference_imgs = []
        reference_img_pixels = []

        obj2frame_mask = {k: top_k for k, v in obj2frame.items()}
        obj2frame_lst = list(obj2frame.values())
        r3scan_bias = 0.5 if dataset == '3rscan' else 1
        blank_img_dim = (240, 320) if dataset == 'scannet' else (224, 172)
        for i, (k, v) in enumerate(obj2frame.items()):  # range(len(obj2frame)):
            obj = v  # obj2frame_lst[i]
            frames, pixels, vis, bbox, pixel_ids = tuple(np.array(t) for t in zip(*obj))

            if dataset == '3rscan':
                bbox = np.array((172-bbox[:, 1], bbox[:, 0], 172-bbox[:, 3], bbox[:, 2])).T
                pixel_ids = np.array([np.array([pids[:, 1], 172-pids[:, 0]]).T for pids in pixel_ids])
                pixel_ids = np.array([pids[(pids[:, 1] < 172) & (pids[:, 0] < 224)] for pids in pixel_ids])

            # filter all where more than 10% of the object is visible
            # self.obj_mask_crit = 0.3
            vis_criterion = ((vis > self.obj_mask_crit) | ((bbox[:, 2]-bbox[:, 0])*(bbox[:, 3]-bbox[:, 1]) > 8000*r3scan_bias))
            if vis_criterion.sum() > 1:
                frames = frames[vis_criterion]
                pixels = pixels[vis_criterion]
                bbox = bbox[vis_criterion]
                vis = vis[vis_criterion]
                pixel_ids = pixel_ids[vis_criterion]
            else:
                blanks = torch.zeros((top_k, *blank_img_dim, 3))
                reference_imgs.append(blanks)
                pixel_ids_img = [np.array([[0, 0]])]*top_k
                reference_img_pixels.append(pixel_ids_img)
                obj2frame_mask[list(obj2frame.keys())[i]] = 0
                continue
                # print('obj not visible enough')

            ids = range(len(vis))
            vis, ids = zip(*sorted(zip(vis, ids), reverse=True))
            frames = frames[np.array(ids)]
            pixel_ids = pixel_ids[np.array(ids)]
            vis, frames, pixel_ids = vis[:top_k], frames[:top_k], pixel_ids[:top_k]
            selected = list(zip(vis, frames, pixel_ids))
            # selected = sorted(zip(vis,frames,pixel_ids))[-top_k:]
            if dataset == 'scannet':
                imgs = [np.asarray(Image.open(os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d", scene_id, "color", s[1])))for s in selected]
            else:
                imgs = [np.asarray(Image.open(os.path.join(CONF.PATH.R3SCAN_RAW, scene_id, 'sequence', s[1])
                                              ).resize((224, 172)).rotate(-90, expand=True)) for s in selected]
            pixel_ids_img = [s[2] for s in selected]

            if not np.array([((img.shape[0] > 0) & (img.shape[1] > 0)) for img in imgs]).all():
                print('zero area')
            processed_imgs = np.stack([im for im in imgs], axis=0)
            # pixel_ids

            if len(imgs) < top_k:
                blanks = np.zeros((top_k-len(imgs), *blank_img_dim, 3))
                processed_imgs = np.concatenate((processed_imgs, blanks), axis=0)

                obj2frame_mask[list(obj2frame.keys())[i]] = len(imgs)
                pixel_ids_img.extend([np.array([[0, 0]])]*(top_k-len(imgs)))

            reference_imgs.append(processed_imgs)
            reference_img_pixels.append(pixel_ids_img)

        return reference_imgs, obj2frame_mask, reference_img_pixels

    def rel_frame_selection(self, rel2frame, rel2frame_mask, scene_id, dataset, top_k=4, scales=2):
        reference_imgs = []
        rel2frame_mask = {k: v*top_k*scales for k, v in rel2frame_mask.items()}
        # rel2frame = list(rel2frame.values())
        r3scan_bias = 0.5 if dataset == '3rscan' else 1
        for i, rel in enumerate(rel2frame.values()):
            # (frame, s_pixels, o_pixels, s_vis, o_vis, s_bbox, o_bbox)
            if len(rel) == 0:
                reference_imgs.append(torch.zeros(((scales)*top_k, 3, self.rel_img_dim, self.rel_img_dim)))
                continue
            frames, s_pixels, o_pixels, s_vis, o_vis, s_bbox, o_bbox = tuple(np.array(t) for t in zip(*rel))

            s_A = (s_bbox[:, 2]-s_bbox[:, 0])*(s_bbox[:, 3]-s_bbox[:, 1])
            o_A = (o_bbox[:, 2]-o_bbox[:, 0])*(o_bbox[:, 3]-o_bbox[:, 1])

            # filter criterion of shared objects
            vis_criterion = ((o_vis > self.rel_vis_crit) | (o_A > 10000*r3scan_bias)
                             ) & ((s_vis > self.rel_vis_crit) | (s_A > 10000*r3scan_bias))

            if vis_criterion.sum() > 1:
                frames = frames[vis_criterion]
                s_bbox = s_bbox[vis_criterion]
                o_bbox = o_bbox[vis_criterion]
                # pixels = s_pixels[vis_criterion] + o_pixels[vis_criterion]
                # share = np.min(np.stack((s_pixels[vis_criterion] / o_pixels[vis_criterion],o_pixels[vis_criterion] / s_pixels[vis_criterion])),axis=0)
                if (np.max(s_pixels[vis_criterion]) - np.min(s_pixels[vis_criterion])) == 0:
                    s_pix_norm = s_pixels[vis_criterion]/s_pixels[vis_criterion]
                else:
                    s_pix_norm = (s_pixels[vis_criterion] - np.min(s_pixels[vis_criterion])) / \
                        (np.max(s_pixels[vis_criterion]) - np.min(s_pixels[vis_criterion]))
                if (np.max(o_pixels[vis_criterion]) - np.min(o_pixels[vis_criterion])) == 0:
                    o_pix_norm = o_pixels[vis_criterion]/o_pixels[vis_criterion]
                else:
                    o_pix_norm = (o_pixels[vis_criterion] - np.min(o_pixels[vis_criterion])) / \
                        (np.max(o_pixels[vis_criterion]) - np.min(o_pixels[vis_criterion]))
                pixels = s_pix_norm+o_pix_norm
                vis = s_vis[vis_criterion] + o_vis[vis_criterion]

            else:
                reference_imgs.append(torch.zeros(((scales)*top_k, 3, self.rel_img_dim, self.rel_img_dim)))
                rel2frame_mask[list(rel2frame.keys())[i]] = 0
                continue

            ids = range(len(vis))
            vis, ids = zip(*sorted(zip(vis, ids), reverse=True))
            frames = frames[np.array(ids)]
            s_bbox = s_bbox[np.array(ids)]
            o_bbox = o_bbox[np.array(ids)]
            vis, frames, s_bbox, o_bbox = vis[:top_k], frames[:top_k], s_bbox[:top_k], o_bbox[:top_k]
            selected = list(zip(vis, frames, s_bbox, o_bbox))

            if dataset == 'scannet':
                imgs = [Image.open(os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d", scene_id, "color", s[1])) for s in selected]
            else:
                imgs = [Image.open(os.path.join(CONF.PATH.R3SCAN_RAW, scene_id, 'sequence', s[1])).resize((224, 172)) for s in selected]

            cropped_imgs = [img.crop(scale_bbox(enclosing_bbox(s[2], s[3]), 1+(sc-1)/2, img))
                            for img, s in zip(imgs, selected) for sc in clip_scales[:scales]]
            if dataset == "3rscan":
                cropped_imgs = [img.rotate(-90, expand=True) for img in cropped_imgs]
            processed_imgs = torch.stack([self.preprocess_rel(im) for im in cropped_imgs], dim=0)

            if len(cropped_imgs) < (scales)*top_k:
                blanks = torch.zeros(((scales)*top_k-len(cropped_imgs), 3, self.rel_img_dim, self.rel_img_dim))
                processed_imgs = torch.cat((processed_imgs, blanks), dim=0)
                rel2frame_mask[list(rel2frame.keys())[i]] = len(cropped_imgs)
            reference_imgs.append(processed_imgs)

        return reference_imgs, rel2frame_mask

    def blip_rel_frames(self, rel2frame, rel2frame_mask, scene_id, dataset, top_k=4, scales=2):
        reference_imgs = []
        rel2frame_mask = {k: v*top_k*scales for k, v in rel2frame_mask.items()}

        blank_img_dim = (320, 240) if dataset == 'scannet' else (224, 172)
        r3scan_bias = 0.5 if dataset == '3rscan' else 1
        # rel2frame = list(rel2frame.values())
        for i, rel in enumerate(rel2frame.values()):
            # (frame, s_pixels, o_pixels, s_vis, o_vis, s_bbox, o_bbox)
            # rel = rel2frame[i]
            if len(rel) == 0:
                black_image = Image.new('RGB', blank_img_dim, (0, 0, 0))
                reference_imgs.append([black_image]*top_k*scales)
                rel2frame_mask[list(rel2frame.keys())[i]] = 0
                continue
            frames, s_pixels, o_pixels, s_vis, o_vis, s_bbox, o_bbox = tuple(np.array(t) for t in zip(*rel))

            # filter all where more than 10% of the object is visible
            s_A = (s_bbox[:, 2]-s_bbox[:, 0])*(s_bbox[:, 3]-s_bbox[:, 1])
            o_A = (o_bbox[:, 2]-o_bbox[:, 0])*(o_bbox[:, 3]-o_bbox[:, 1])

            vis_criterion = ((o_vis > self.rel_vis_crit/2) | (o_A > 10000*r3scan_bias)
                             ) & ((s_vis > self.rel_vis_crit/2) | (s_A > 10000*r3scan_bias))

            if vis_criterion.sum() > 1:
                frames = frames[vis_criterion]
                s_bbox = s_bbox[vis_criterion]
                o_bbox = o_bbox[vis_criterion]

                if (np.max(s_pixels[vis_criterion]) - np.min(s_pixels[vis_criterion])) == 0:
                    s_pix_norm = s_pixels[vis_criterion]/s_pixels[vis_criterion]
                else:
                    s_pix_norm = (s_pixels[vis_criterion] - np.min(s_pixels[vis_criterion])) / \
                        (np.max(s_pixels[vis_criterion]) - np.min(s_pixels[vis_criterion]))
                if (np.max(o_pixels[vis_criterion]) - np.min(o_pixels[vis_criterion])) == 0:
                    o_pix_norm = o_pixels[vis_criterion]/o_pixels[vis_criterion]
                else:
                    o_pix_norm = (o_pixels[vis_criterion] - np.min(o_pixels[vis_criterion])) / \
                        (np.max(o_pixels[vis_criterion]) - np.min(o_pixels[vis_criterion]))
                pixels = s_pix_norm+o_pix_norm
                vis = s_vis[vis_criterion] + o_vis[vis_criterion]

            else:
                black_image = Image.new('RGB', blank_img_dim, (0, 0, 0))
                reference_imgs.append([black_image]*top_k*scales)
                rel2frame_mask[list(rel2frame.keys())[i]] = 0
                continue

            ids = range(len(vis))
            vis, ids = zip(*sorted(zip(vis, ids), reverse=True))
            frames = frames[np.array(ids)]
            s_bbox = s_bbox[np.array(ids)]
            o_bbox = o_bbox[np.array(ids)]
            vis, frames, s_bbox, o_bbox = vis[:top_k], frames[:top_k], s_bbox[:top_k], o_bbox[:top_k]
            selected = list(zip(vis, frames, s_bbox, o_bbox))

            if dataset == 'scannet':
                imgs = [Image.open(os.path.join(CONF.PATH.SCANNET_RAW, "scannet_2d", scene_id, "color", s[1])) for s in selected]
            else:
                imgs = [Image.open(os.path.join(CONF.PATH.R3SCAN_RAW, scene_id, 'sequence', s[1])) for s in selected]
            rel2frame_mask[list(rel2frame.keys())[i]] = len(imgs)

            imgs = [img.crop(scale_bbox(enclosing_bbox(s[2], s[3]), 1+(sc-1)/2, img))
                    for img, s in zip(imgs, selected) for sc in clip_scales[:scales]]

            if dataset == "3rscan":
                imgs = [img.rotate(-90, expand=True) for img in imgs]

            if len(imgs) < (scales)*top_k:
                black_image = Image.new('RGB', blank_img_dim, (0, 0, 0))
                imgs.extend([black_image]*((top_k*scales)-len(imgs)))

            reference_imgs.append(imgs)

        return reference_imgs, rel2frame_mask

    def load_imgs(self, data_dict):
        obj_imgs, obj2frame_mask = self.obj_frame_selection(
            data_dict["obj2frame"], data_dict["scene_id"], data_dict['dataset'], top_k=self.top_k_frames, scales=self.scales)
        obj_imgs = torch.stack(obj_imgs, dim=0)
        data_dict["object_imgs"] = torch.zeros((self.max_objs, *obj_imgs.shape[1:]))
        data_dict["object_imgs"][:len(obj_imgs)] = obj_imgs

        data_dict['obj2frame_mask'] = np.array(list(obj2frame_mask.values())).astype(float)
        padding_width = ((0, self.max_objs - data_dict["obj2frame_mask"].shape[0]),)
        data_dict["obj2frame_mask"] = np.pad(data_dict["obj2frame_mask"], padding_width, mode='constant', constant_values=0)

        if self.openseg:
            obj_raw_imgs, obj2frame_raw_mask, obj_frame_pixels = self.obj_pixel_selection(
                data_dict["obj2frame"], data_dict["scene_id"], data_dict['dataset'], top_k=self.top_k_frames)
            obj_raw_imgs = np.stack(obj_raw_imgs, axis=0)
            data_dict["object_raw_imgs"] = np.zeros((self.max_objs, *obj_raw_imgs.shape[1:]))
            data_dict["object_raw_imgs"][:len(obj_raw_imgs)] = obj_raw_imgs

            data_dict['obj2frame_raw_mask'] = np.array(list(obj2frame_raw_mask.values())).astype(float)
            padding_width = ((0, self.max_objs - data_dict["obj2frame_raw_mask"].shape[0]),)
            data_dict["obj2frame_raw_mask"] = np.pad(data_dict["obj2frame_raw_mask"], padding_width, mode='constant', constant_values=0)

            data_dict['object_pixels'] = obj_frame_pixels

        if self.blip or self.llava:
            rel_imgs, rel2frame_mask = self.blip_rel_frames(
                data_dict["rel2frame"], data_dict['rel2frame_mask'], data_dict["scene_id"], data_dict['dataset'], top_k=self.top_k_frames, scales=self.scales)
            blank_img_dim = (320, 240) if data_dict['dataset'] == 'scannet' else (224, 172)
            black_image = Image.new('RGB', blank_img_dim, (0, 0, 0))
            rel_imgs.extend([[black_image]*self.top_k_frames*self.scales]*(self.max_rels-len(rel_imgs)))
            data_dict['blip_images'] = rel_imgs

            data_dict['blip_mask'] = rel2frame_mask

            data_dict['rel2frame_mask'] = np.array(list(rel2frame_mask.values())).astype(float)
            padding_width = ((0, self.max_rels - data_dict["rel2frame_mask"].shape[0]),)
            data_dict["rel2frame_mask"] = np.pad(data_dict["rel2frame_mask"], padding_width, mode='constant', constant_values=0)
            data_dict["rel2frame_mask"] = data_dict["rel2frame_mask"]
            data_dict["relationship_imgs"] = []
        else:
            rel_imgs, rel2frame_mask = self.rel_frame_selection(
                data_dict["rel2frame"], data_dict['rel2frame_mask'], data_dict["scene_id"], data_dict['dataset'], top_k=self.top_k_frames, scales=self.scales)

            rel_imgs = torch.stack(rel_imgs, dim=0)
            data_dict["relationship_imgs"] = torch.zeros((self.max_rels, *rel_imgs.shape[1:]))
            data_dict["relationship_imgs"][:len(rel_imgs)] = rel_imgs

            data_dict['rel2frame_mask'] = np.array(list(rel2frame_mask.values())).astype(float)
            padding_width = ((0, self.max_rels - data_dict["rel2frame_mask"].shape[0]),)
            data_dict["rel2frame_mask"] = np.pad(data_dict["rel2frame_mask"], padding_width, mode='constant', constant_values=0)
            data_dict["rel2frame_mask"] = data_dict["rel2frame_mask"]

        return data_dict

    def load_features_disk(self, data_dict):
        scan_id = data_dict['scan_id']
        obj_valid_rel_path = sorted(os.listdir(self.load_features))
        assert len(obj_valid_rel_path) == 3

        torch_numpy = '.pt'
        obj_feature_pth = os.path.join(self.load_features, obj_valid_rel_path[0], scan_id+torch_numpy)
        obj_valid_feature_pth = os.path.join(self.load_features, obj_valid_rel_path[1], scan_id+torch_numpy)
        rel_feature_pth = os.path.join(self.load_features, obj_valid_rel_path[2], scan_id+torch_numpy)

        if torch_numpy == '.npy':
            obj_features = torch.from_numpy(np.load(obj_feature_pth))
            obj_valid_features = torch.from_numpy(np.load(obj_valid_feature_pth))
            rel_features = torch.from_numpy(np.load(rel_feature_pth))
        else:
            obj_features = torch.load(obj_feature_pth)
            obj_valid_features = torch.load(obj_valid_feature_pth)
            rel_features = torch.load(rel_feature_pth)

        data_dict["clip_obj_encoding"] = torch.zeros((self.max_objs, obj_features.shape[-1]))
        data_dict["clip_obj_encoding"][:len(obj_features)] = obj_features
        data_dict["clip_obj_valids"] = torch.zeros((self.max_objs), dtype=torch.bool)
        data_dict["clip_obj_valids"][:len(obj_valid_features)] = obj_valid_features
        if self.blip or self.llava:
            data_dict['clip_rel_encoding'] = torch.zeros((self.max_rels, *rel_features.shape[-2:]), dtype=rel_features.dtype)
        else:
            data_dict['clip_rel_encoding'] = torch.zeros((self.max_rels, rel_features.shape[-1]))
        data_dict["clip_rel_encoding"][:len(rel_features)] = rel_features
        return data_dict

    def __getitem__(self, idx):
        # start = time.time()

        data_dict = vars(self.scene_data[idx])
        padding_width = ((0, self.max_objs - data_dict["objects_id"].shape[0]),)
        data_dict["objects_id"] = np.pad(data_dict["objects_id"].astype(float), padding_width, mode='constant', constant_values=0)
        data_dict["objects_cat"] = np.pad(data_dict["objects_cat"].astype(float), padding_width, mode='constant', constant_values=0)
        data_dict["objects_num"] = np.pad(data_dict["objects_num"].astype(float), padding_width, mode='constant', constant_values=0)

        data_dict["objects_scale"] = np.pad(data_dict["objects_scale"], padding_width, mode='constant', constant_values=0)
        data_dict["objects_scale"] = data_dict["objects_scale"].astype(np.float32)

        padding_width = ((0, self.max_objs - data_dict["objects_pcl"].shape[0]), (0, 0), (0, 0))
        data_dict["objects_pcl"] = np.pad(data_dict["objects_pcl"], padding_width, mode='constant')
        data_dict["objects_pcl"] = data_dict["objects_pcl"].astype(np.float32)

        padding_width = ((0, self.max_objs - data_dict["objects_center"].shape[0]), (0, 0))
        data_dict["objects_center"] = np.pad(data_dict["objects_center"], padding_width, mode='constant', constant_values=0)
        data_dict["objects_center"] = data_dict["objects_center"].astype(np.float32)

        padding_width = ((0, self.max_rels - data_dict["predicate_cat"].shape[0]), (0, 0))
        data_dict["predicate_cat"] = np.pad(data_dict["predicate_cat"], padding_width, mode='constant', constant_values=0)
        data_dict["predicate_cat"] = data_dict["predicate_cat"].astype(np.int64)

        data_dict["predicate_dist"] = np.pad(data_dict["predicate_dist"], padding_width, mode='constant', constant_values=0)
        data_dict["predicate_dist"] = data_dict["predicate_dist"].astype(np.float32)
        data_dict["predicate_dist"][np.isnan(data_dict["predicate_dist"])] = 0.

        if "predicate_min_dist" in data_dict.keys():
            data_dict["predicate_min_dist"] = np.pad(data_dict["predicate_min_dist"], padding_width, mode='constant', constant_values=0)
            data_dict["predicate_min_dist"] = data_dict["predicate_min_dist"].astype(np.float32)
            data_dict["predicate_min_dist"][np.isnan(data_dict["predicate_min_dist"])] = 0.

        padding_width = ((0, self.max_rels - data_dict["predicate_num"].shape[0]))
        data_dict["predicate_num"] = np.pad(data_dict["predicate_num"].astype(float), padding_width, mode='constant', constant_values=0)
        # data_dict["predicate_num"] = data_dict["predicate_num"].astype(np.int64)

        padding_width = ((0, self.max_rels - data_dict["predicate_pcl_flag"].shape[0]), (0, 0), (0, 0))
        data_dict["predicate_pcl_flag"] = np.pad(data_dict["predicate_pcl_flag"], padding_width, mode='constant')
        data_dict["predicate_pcl_flag"] = data_dict["predicate_pcl_flag"].astype(np.float32)

        padding_width = ((0, self.max_rels - data_dict["pairs"].shape[0]), (0, 0))
        data_dict["pairs"] = np.pad(data_dict["pairs"].astype(float), padding_width, mode='constant', constant_values=0)
        data_dict["pairs"] = data_dict["pairs"].astype(np.int64)

        data_dict["edges"] = np.pad(data_dict["edges"].astype(float), padding_width, mode='constant', constant_values=0)
        data_dict["edges"] = data_dict["edges"].astype(np.int64)

        # data_dict["triples"] = np.pad(data_dict["triples"].astype(float), ((0, 200 - data_dict["predicate_cat"].shape[0]),(0,0)) , mode='constant',constant_values=0)
        data_dict["triples"] = data_dict["triples"].astype(np.int64)

        if not self.load_features:
            data_dict = self.load_imgs(data_dict)
        else:
            data_dict = self.load_features_disk(data_dict)

        if data_dict['dataset'] == "scannet":
            # scannet uses axes aligned bboxes
            objs_bbox = np.array(data_dict["tight_bbox"])
        else:
            objs_bbox = np.array([data_dict["tight_bbox"][i]['axis_aligned'] for i in range(len(data_dict["tight_bbox"]))])

        data_dict["objects_bbox"] = np.zeros((self.max_objs, 7), dtype=np.float32)
        data_dict["objects_bbox"][:len(objs_bbox)] = objs_bbox

        if data_dict['dataset'] == '3rscan':
            del data_dict['objects_scale']
            predicate_edges = []
            for edge in torch.tensor(data_dict['predicate_cat']):
                predicates = list(np.where(edge == 1)[0])
                predicate_edges.append(predicates)
            data_dict["predicate_edges"] = predicate_edges
            predicate_edges_len = [len(e) for e in predicate_edges]
            edge_mapping = list(accumulate(predicate_edges_len))
            edge_mapping = [0] + edge_mapping
            data_dict["edge_mapping"] = edge_mapping

        return data_dict

    def collate_fn(self, data):
        data_dict = {}
        keys = data[0].keys()

        for key in keys:
            if key == 'triples':
                data_dict[key] = [torch.from_numpy(data[i][key]) for i in range(len(data))]
            elif type(data[0][key]) is str:
                data_dict[key] = [data[i][key] for i in range(len(data))]
            elif type(data[0][key]) is list:
                data_dict[key] = [data[i][key] for i in range(len(data))]
            elif type(data[0][key]) is torch.Tensor:
                data_dict[key] = torch.stack([data[i][key] for i in range(len(data))])
            elif type(data[0][key]) is np.ndarray:
                data_dict[key] = torch.from_numpy(np.stack([data[i][key] for i in range(len(data))]))
                # data_dict[key] = np.zeros((len(9,*data[1][key].shape[1:]))
            elif type(data[0][key]) is dict:
                if key == "id2name":
                    data_dict[key] = [data[i][key] for i in range(len(data))]
            else:
                raise

        data_dict["aligned_obj_num"] = torch.tensor(self.max_objs)
        data_dict["aligned_rel_num"] = torch.tensor(self.max_rels)

        data_dict["objects_bbox"][:, :, -1] = data_dict["objects_bbox"][:, :, -1] * ((2*np.pi)/360)
        data_dict["objects_bbox"][:, :, -1] = data_dict["objects_bbox"][:, :, -1] % (2*np.pi)

        data_dict["objects_pcl"][..., 3:6] = data_dict["objects_pcl"][..., 3:6]/255
        data_dict["predicate_pcl_flag"][..., 3:6] = data_dict["predicate_pcl_flag"][..., 3:6]/255

        return data_dict


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from util.plotting_utils import *

    scans = json.load(open(os.path.join(CONF.PATH.SCANNET, "relationships_validation_filter_wo_misaligned.json")))["scans"]
    scans_r3scan = json.load(open(os.path.join(CONF.PATH.R3SCAN_RAW, "relationships_validation.json")))["scans"]
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors_hex = list(mcolors.TABLEAU_COLORS.values())

    dataset = Open2D3DSGDataset(relationships_scannet=None, relationships_R3SCAN=scans_r3scan, mini=True,
                                img_dim=336, top_k_frames=3, scales=2,
                                rel_texts=False, blip=False, openseg=True)

    start = time.time()
    for i in tqdm(range(len(dataset.scene_data)-1)):
        data = dataset.__getitem__(i)
        data_collate = dataset.collate_fn([data, dataset.__getitem__(i+1)])
        # data_collate = dataset.collate_fn([data])
        # if data_collate['scan_id'] == 'scene0011_00-0':
        #     break
        # data_collate = dataset.collate_fn([data])

    # data_collate = dataset.collate_fn([dataset.__getitem__(random.randint(0,20))])
    print((time.time()-start)/(len(dataset.scene_data)-1))
    print(data_collate['scan_id'])
    bboxes = data_collate['objects_bbox'][0]
    shapes = data_collate['objects_pcl'][0]
    shapes_pair = data_collate['predicate_pcl_flag'][0]
    corners = np.zeros((data_collate['objects_cat'][0].shape[0], 8, 3))

    obj_ids = data_collate['objects_id'][0][0]
    obj_imgs = data_collate['object_imgs'][0][0]  # batch, node, view+scale, c, w, h
    # obj_imgposes = data_collate['object_imgposes'][0][0] #batch, node, view+scale, 4, 4
    obj_shape = data_collate['objects_pcl'][0][0]
    rel_ids = data_collate['objects_id'][0][data_collate['edges'][0]][0]
    rel_imgs = data_collate['relationship_imgs'][0][0]  # batch, pair, view+scale, c, w, h
    rel_shape = data_collate['predicate_pcl_flag'][0][0]

    # scan_info = [s for s in scans if (s['scan']==data_collate['scan_id'].split('-')[0] and s['split']==int(data_collate['scan_id'].split('-')[1]))]
    print(data_collate['scan_id'])

    pcls = []
    colors_points = []
    corner_points = []
    for i, shape in enumerate(shapes):

        corners[i] = params_to_8points_torch(bboxes[i][None])
        corners_ = corners[i]

        corner_points.append(corners_)
        verts = vertify(corners_)

        pcl = fit_shapes_to_box(bboxes[i][:6], shape[:, :3], withangle=False)
        # if i==0:
        #     pcl[...,2] = pcl[...,2]-2

        color = matplotlib.colors.to_rgb(colors_hex[i])
        # ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors="black", alpha=.00))
        pcls.append(pcl)
        colors_points.append(np.tile(color, (pcl.shape[0], 1)))
    pcls = np.concatenate(pcls, axis=0)
    colors_points = np.concatenate(colors_points, axis=0)
    corner_points = np.concatenate(corner_points, axis=0)
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.scatter(pcls[:, 0], pcls[:, 1], pcls[:, 2], c=colors_points, marker=".")
    set_axes_equal(ax)
    for i in range(len(obj_imgs)):
        f = plt.figure()
        ax = f.add_subplot(231, projection='3d')

        ax.scatter(pcls[:, 0], pcls[:, 1], pcls[:, 2], c=colors_points, marker=".")

        set_axes_equal(ax)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.axis('off')
        # plt.grid(b=None)
        ax = f.add_subplot(232, projection='3d')
        ax.scatter(obj_shape[:, 0], obj_shape[:, 1], obj_shape[:, 2], c=obj_shape[:, 3:6])
        set_axes_equal(ax)
        ax = f.add_subplot(233)
        ax.imshow(unnormalize_image(obj_imgs[i].permute(1, 2, 0).numpy()))
        # ax.set_title(scan_info[0]['objects'][str(int(obj_ids.item()))])
        ax = f.add_subplot(235, projection='3d')
        ax.scatter(rel_shape[:, 0], rel_shape[:, 1], rel_shape[:, 2], c=rel_shape[:, -1])
        set_axes_equal(ax)
        ax = f.add_subplot(236)
        ax.imshow(unnormalize_image(rel_imgs[i].permute(1, 2, 0).numpy()))
        # ax.set_title(','.join([scan_info[0]['objects'][str(int(rel_ids[0].item()))],scan_info[0]['objects'][str(int(rel_ids[1].item()))]]))
        f.suptitle(data_collate['scan_id'])

    plt.show()
