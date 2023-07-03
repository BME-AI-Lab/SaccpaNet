# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.codec import MSRAHeatmap
from .utils.transforms import affine_transform, fliplr_joints, get_affine_transform

logger = logging.getLogger(__name__)

from configs.dataset_config import *


class JointsDataset(Dataset):
    def __init__(self, is_train, transform=None):
        # root, image_set
        self.setup_config_constants()
        self.num_joints = 0
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train

        self.flip = self.is_train  # enable joint flip as data augmentation for training

        self.transform = transform
        self.db = []

        self.heatmap_generator = MSRAHeatmap(
            self.image_size,
            self.heatmap_size,
            sigma=self.sigma,
            unbiased=True,
            # blur kernel size is not used for biased heatmap
        )

    def setup_config_constants(self):
        # image size
        self.image_size = IMAGE_SIZE
        # heatmap configs
        self.target_type = TARGET_TYPE
        self.heatmap_size = HEATMAP_SIZE
        self.sigma = SIGMA  # sigma for the gaussian distribution

        # transformation config
        self.scale_factor = SCALE_FACTOR
        self.rotation_factor = ROTATION_FACTOR

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def _get_sql_image_connections(self):
        raise NotImplementedError

    def _get_numpy_image(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        imgnum = db_rec["imgnum"] if "imgnum" in db_rec else ""

        if self.data_format == "sql":
            data_numpy = self._get_numpy_image(idx)

        else:
            assert False, "Need to define data_format"

        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints = db_rec["joints_3d"]
        joints_vis = db_rec["joints_3d_vis"]

        center = db_rec["center"]
        scale = db_rec["scale"]
        score = db_rec["score"] if "score" in db_rec else 1
        rotation = 0

        # TBD: handle by co-transform
        flipped = False
        if self.is_train:
            scale, rotation = self.calculate_random_scale_and(scale)

            if self.flip and random.random() <= 0.5:
                data_numpy, joints, joints_vis = self.flip_joints(
                    joints, joints_vis, data_numpy, center
                )
                flipped = True

        trans = get_affine_transform(center, scale, rotation, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR,
        )
        # Support for post transformation of image
        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            # if joints_vis[i, 0] > 0.0:
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        heatmap, heatmap_weight = self.generate_heatmap(joints, joints_vis)

        heatmap = torch.from_numpy(heatmap)
        heatmap_weight = torch.from_numpy(heatmap_weight)
        meta = {
            "image": image_file,
            "filename": filename,
            "imgnum": imgnum,
            "joints": joints,
            "joints_vis": joints_vis,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "score": score,
            "flipped": flipped,
        }

        return input, heatmap, heatmap_weight, meta

    def flip_joints(self, joints, joints_vis, data_numpy, c):
        data_numpy = data_numpy[:, ::-1]  # 3 channels image only
        joints, joints_vis = fliplr_joints(
            joints, joints_vis, data_numpy.shape[1], self.flip_pairs
        )
        c[0] = data_numpy.shape[1] - c[0] - 1
        return data_numpy, joints, joints_vis

    def calculate_random_scale_and(self, s):
        sf = self.scale_factor
        rf = self.rotation_factor
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = (
            np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
            if random.random() <= 0.6
            else 0
        )
        return s, r

    def generate_heatmap(self, joints, joints_vis):
        # expand dims for heatmap generator to work
        joints = np.expand_dims(joints[:, :2], axis=0)
        joints_vis = np.expand_dims(joints_vis[:, 0], axis=0)
        heatmap_dict = self.heatmap_generator.encode(joints, joints_vis)
        heatmaps = heatmap_dict["heatmaps"]
        # add dummy z axis back
        heatmap_weights = heatmap_dict["keypoint_weights"]
        heatmap_weights = heatmap_weights.reshape((self.num_joints, 1))

        return heatmaps, heatmap_weights
