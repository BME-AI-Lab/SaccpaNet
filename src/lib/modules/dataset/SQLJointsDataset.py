import logging

import numpy as np
import pandas as pd

from lib.modules.dataset.common import ResolveImage
from lib.modules.dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)
from random import random, randrange, seed

seed(42)
import functools

from configs.dataset_config import DB_CONNECTION_STRING


class SQLJointsDataset(JointsDataset):
    DB_CONNECTION_STRING = DB_CONNECTION_STRING
    TABLE_NAME = "annotations"

    def __init__(
        self, is_train, transform=None, mixed=True, all_quilt=False, test=True
    ):
        super().__init__(is_train, transform)
        if is_train:
            self.subset = "train"
        elif test:
            self.subset = "test"
        else:
            self.subset = "val"
        self.num_joints = 18
        indexs = self._get_sql_indexs()
        self._initDB(self.subset, indexs)
        self.pixel_std = 200  # 200 ?? TBD
        self.data_format = "sql"
        # image_widt
        self.image_width = self.image_size[0]  # cfg.MODEL.IMAGE_SIZE[0] # TBD
        self.image_height = self.image_size[1]  # cfg.MODEL.IMAGE_SIZE[1] #TBD
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.db = self._get_db()
        logger.info("=> load {} samples".format(len(self.db)))
        image_database = "depth_images"
        self.image_query_string = (
            f"SELECT a.depth_array FROM {image_database}" + " as a WHERE a.`index`={}"
        )
        self.image_indexer = None
        if mixed:
            self.probability = 0.5
        else:
            self.probability = False
        self.all_quilt = all_quilt

    def _initDB(self, subset, indexs):
        self.db_connection_string = self.DB_CONNECTION_STRING
        self.table_name = self.TABLE_NAME
        self.pose = pd.read_sql_query(
            "SELECT "
            + indexs
            + f" FROM {self.table_name} AS a WHERE a.subset='{subset}'",
            self.db_connection_string,
        )
        self.annotations_df = pd.read_sql_query(
            f"SELECT * FROM {self.table_name} AS a WHERE a.subset='{subset}'",
            self.db_connection_string,
        )

    def _get_sql_indexs(self):
        num_index = self.num_joints
        indexs = ["a.`index`"]
        for i in range(num_index):
            indexs.append(f"a.`{i}_x`, a.`{i}_y`")
        return ", ".join(indexs)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    @property
    @functools.lru_cache()
    def _get_posture_lookup_dict(self):
        lookup_table = dict()
        for idx in range(len(self.annotations_df)):
            row = self.annotations_df.iloc[idx]
            # print(row)
            lookup_table[(row["source_file"], row["posture"], row["effect"])] = idx
        return lookup_table

    def _get_numpy_image(self, idx):
        indexer = self._get_sql_image_connections()
        # print(self.annotations_df.iloc[idx])
        row = self.annotations_df.iloc[idx]
        source_file = row["source_file"]
        posture = row["posture"]
        effect = row["effect"]
        # print(self.annotations_df.iloc[idx])
        if self.subset == "test" or random() > self.probability:
            data_numpy = indexer(self.annotations_df.iloc[idx])
        else:
            # print(idx)
            data_numpy1 = indexer(self.annotations_df.iloc[idx])
            random_effect = str(randrange(1, 5))
            # print(self._get_posture_lookup_dict)
            random_idx = self._get_posture_lookup_dict[
                (source_file, posture, random_effect)
            ]
            data_numpy2 = indexer(self.annotations_df.iloc[random_idx])
            merge_probability = random()
            data_numpy = (
                merge_probability * data_numpy1 + (1 - merge_probability) * data_numpy2
            )
        return data_numpy

    def _get_sql_image_connections(self):
        # cache getter
        if self.image_indexer:
            return self.image_indexer
        assert self.image_query_string
        assert self.db_connection_string

        self.image_indexer = ResolveImage(
            self.image_query_string, self.db_connection_string, resize=None
        )  # (192,256)
        return self.image_indexer

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        # fix_aspeck ratio
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32
        )
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _get_db(self):
        dataset = self
        # print(dataset.pose)
        indexs = dataset.pose.iloc[:, 0]
        # print(indexs)
        joints = np.array(dataset.pose.iloc[:, 1:])
        # remark xs,ys swaped due to upright
        xs, ys = (
            joints[:, 0::2],
            joints[:, 1::2],
        )  # TODO: check if it is the case OR place assertion
        zeros = np.zeros_like(xs)
        joints_3d = np.stack([xs, ys, zeros], axis=2)
        # to be determined for visibility
        _vis = np.int32((xs != 0.0) & (ys != 0.0))
        joints_3d_vis = np.stack([_vis, _vis, zeros], axis=2)

        # boudning box by joint location
        # warn x,y swapped due?
        xs_min = np.min(xs, axis=1)
        ys_min = np.min(ys, axis=1)
        xs_max = np.max(xs, axis=1)
        ys_max = np.max(ys, axis=1)
        bboxs = np.stack([xs_min, ys_min, xs_max - xs_min, ys_max - ys_min], axis=1)
        # posture group mapping

        rec = []
        for i in range(len(dataset.pose)):
            index = indexs[i]
            center, scale = self._box2cs(bboxs[i, :4])
            rec.append(
                {
                    "image": index,  # self.image_path_from_index(index),
                    "center": center,
                    "scale": scale,
                    "joints_3d": joints_3d[i],
                    "joints_3d_vis": joints_3d_vis[i],
                    "filename": "",
                    "imgnum": 0,  # ?
                    # missing score
                }
            )
            # print(rec[:-1])
        return rec

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return super().evaluate(cfg, preds, output_dir, *args, **kwargs)

    def __getitem__(self, idx):
        input, target, target_weight, meta = super().__getitem__(idx)
        posture_groups = dict()
        posture_groups[2] = {"r": 0, "o": 1, "y": 1, "g": 1, "c": 1, "b": 0, "p": 0}
        posture_groups[4] = {"r": 0, "o": 1, "y": 1, "g": 2, "c": 2, "b": 3, "p": 3}
        posture_groups[6] = {"r": 0, "o": 1, "y": 2, "g": 3, "c": 4, "b": 5, "p": 5}
        posture_groups[7] = {"r": 0, "o": 1, "y": 2, "g": 3, "c": 4, "b": 5, "p": 6}
        flip_symbol = {
            "r": "r",
            "o": "g",
            "y": "c",
            "g": "o",
            "c": "y",
            "b": "p",
            "p": "b",
        }
        posture_symbol = self.annotations_df.posture.iloc[idx]
        effect_symbol = self.annotations_df.effect.iloc[idx]
        meta["effect"] = effect_symbol
        if "flipped" in meta and meta["flipped"]:
            posture_symbol = flip_symbol[posture_symbol]
        meta["posture"] = posture_groups[7][posture_symbol]
        input = np.expand_dims(input, 0)
        return input, target, target_weight, meta


if __name__ == "__main__":
    """Test the dataset"""
    dataset = SQLJointsDataset(is_train=True)
    db = dataset._get_db()
    ds_iter = iter(dataset)
    a, b, c, d = next(ds_iter)
    import matplotlib.pyplot as plt

    # test print joint
    a, b, c, d = next(ds_iter)
    plt.imshow(a[0])
    plt.show()
    zeros = np.zeros_like(b[0])
    for i in range(18):
        zeros = zeros + b[i].numpy()
        # plt.imshow(b[i])
        # plt.show()
    plt.imshow(zeros)
    plt.show()
