# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
from torch import nn


class PclAugmenter(nn.Module):
    def __init__(self, prob, jitter=True, scale=True, rotate=True, rgb_jitter=True):
        super().__init__()
        self.prob = prob
        self.jitter = jitter
        self.scale = scale
        self.rotate = rotate
        self.rgb_jitter = rgb_jitter

    @torch.no_grad()
    def forward(self, data_dict):
        device = data_dict['objects_id'].device
        objects = data_dict['objects_pcl']
        predicates = data_dict['predicate_pcl_flag']

        # TODO: calculate mask of zeros which should stay zero
        obj_mask = (objects != 0).all(axis=2)
        pred_mask = (predicates != 0).all(axis=2)
        # jitter
        if np.random.uniform() < self.prob:
            objects[:, :, :3][obj_mask] += torch.normal(0, 0.01, size=objects[:, :, :3].shape)[obj_mask].to(device)
            predicates[:, :, :3][pred_mask] += torch.normal(0, 0.01, size=predicates[:, :, :3].shape)[pred_mask].to(device)

        # scale
        if np.random.uniform() < self.prob:
            objects[:, :, :3][obj_mask] *= torch.from_numpy(np.random.uniform(0.9, 1.1,
                                                            size=objects.shape[:2])[..., None])[obj_mask].to(device)
            predicates[:, :, :3][pred_mask] *= torch.from_numpy(np.random.uniform(0.9,
                                                                1.1, size=predicates.shape[:2])[..., None])[pred_mask].to(device)

        # rotate (objects only)
        if np.random.uniform() < self.prob:
            theta = np.random.uniform(0, 2*np.pi)
            R_z = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]]).float()
            objects[:, :, :3][obj_mask] = torch.matmul(objects[:, :, :3], R_z.to(device))[obj_mask]

        # rgb jitter
        if np.random.uniform() < self.prob:
            objects[:, :, 3:6][obj_mask] += torch.from_numpy(np.random.uniform(-0.02, 0.02,
                                                             size=objects[:, :, 3:6].shape))[obj_mask].to(device)
            predicates[:, :, 3:6][pred_mask] += torch.from_numpy(np.random.uniform(-0.02,
                                                                 0.02, size=predicates[:, :, 3:6].shape))[pred_mask].to(device)

        data_dict['objects_pcl'] = objects
        data_dict['predicates_pcl'] = predicates

        return data_dict


def jittering(data, prob):
    if np.random.uniform() < prob:
        data[:, :, :3] += np.random.normal(0, 0.01, size=data[:, :, :3].shape)

    return data


def scaling(data, prob):
    if np.random.uniform() < prob:
        data[:, :, :3] *= np.random.uniform(0.9, 1.1, size=data.shape[:2])[..., None]
    return data


def rotating(data, prob):
    if np.random.uniform() < prob:
        theta = np.random.uniform(0, 2*np.pi)
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        data[:, :, :3] = np.matmul(data[:, :, :3], R_z)
    return data


def rgb_jittering(data, prob):
    if np.random.uniform() < prob:
        data[:, 3:] += np.random.randint(-5, 5, size=data[:, :, 3:].shape)
    return data


def rgb_light_effect(data, prob):
    if np.random.uniform() < prob:
        data[:, 3:] += np.random.normal(0, 0.07, size=data[:, :, 3:].shape)
        data[:, 3:] = np.clip(data[:, 3:], 0, 255)
    return data


if __name__ == "__main__":
    pcl = np.random.rand(8, 100, 3)
    rotating(pcl, 1.0)
