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

from .codec import MSRAHeatmap
from .utils.transforms import (affine_transform, fliplr_joints,
                               get_affine_transform)

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, is_train, transform=None):
        # root, image_set
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train

        # self.root = root
        # self.image_set = image_set

        # as one off, instead of loading from cfg, hot patch
        # referencing cfg

        # self.output_path = cfg.OUTPUT_DIR # not used
        self.data_format = "sql"  # cfg.DATASET.DATA_FORMAT # not used

        # self.scale_factor = cfg.DATASET.SCALE_FACTOR # TBD:should be handled by co-transform
        # self.rotation_factor = cfg.DATASET.ROT_FACTOR # TBD:should be handled by co-transform
        # self.flip = cfg.DATASET.FLIP

        self.image_size = np.array((192, 256))  # (192,256) #cfg.MODEL.IMAGE_SIZE
        self.target_type = "gaussian"  # cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = np.array((24, 32))
        self.sigma = 2  # 2#cfg.MODEL.EXTRA.SIGMA

        self.scale_factor = 0.3
        self.flip = self.is_train
        self.rotation_factor = 40

        self.transform = transform
        self.db = []

        self.heatmap_generator = MSRAHeatmap(
            self.image_size,
            self.heatmap_size,
            sigma=self.sigma,
            unbiased=False,
            # blur kernel size is not used for biased heatmap
        )

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(
        self,
    ):
        return len(self.db)

    def _get_sql_image_connections(self):
        raise NotImplementedError

    def _get_numpy_image(self, idx):
        indexer = self._get_sql_image_connections()
        data_numpy = indexer(self.annotations_df.iloc[idx])
        return data_numpy

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

        trans = get_affine_transform(
            center, scale, rotation, self.image_size
        )  # disabled for co-transfomr
        #!!!Doubtable for cv2.warpAffine in float!!!
        # """
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
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
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

        return input, target, target_weight, meta

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

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(rec["joints_3d"], rec["joints_3d_vis"]):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec["scale"][0] * rec["scale"][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec["center"])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2**2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info("=> num db: {}".format(len(db)))
        logger.info("=> num selected db: {}".format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        # expand dims for heatmap generator to work
        joints = np.expand_dims(joints[:, :2], axis=0)
        joints_vis = np.expand_dims(joints_vis[:, 0], axis=0)
        heatmap_dict = self.heatmap_generator.encode(joints, joints_vis)
        heatmaps = heatmap_dict["heatmaps"]
        # add dummy z axis back
        # heatmaps = np.expand_dims(heatmaps, axis=0)
        # h, w, _ = heatmaps.shape
        # heatmaps = np.concatenate([heatmaps, np.zeros((n, h, w, 1))], axis=3)
        heatmap_weights = heatmap_dict["keypoint_weights"]
        heatmap_weights = heatmap_weights.reshape((self.num_joints, 1))

        return heatmaps, heatmap_weights
