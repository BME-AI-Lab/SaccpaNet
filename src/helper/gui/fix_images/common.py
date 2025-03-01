import pickle
from collections import namedtuple
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
import sqlalchemy

_lock = Lock()


class SQL_indexer:
    def __init__(self, query_string, connection):
        self.engine = sqlalchemy.create_engine(connection, echo=False)
        self.query_string = query_string

    @lru_cache(maxsize=200000)
    def __getitem__(self, key):
        self.conn = self.engine.connect()
        _lock.acquire()
        result = self.conn.execute(self.query_string.format(key))
        assert result.rowcount == 1
        initial_row = result.first()
        assert len(initial_row.keys()) == 1
        _lock.release()
        return initial_row[0]  # result[0][0]


def fill_hole(img):
    img[img == 0] = np.median(img)
    img[img > 3 * 1000] = np.median(img)
    return img


class ResolveImage:
    def __init__(
        self,
        query_string="SELECT a.depth_array FROM data_06_15_images as a WHERE a.index={}",
        connection="mysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu",
        resize=(224, 224),
    ):
        self.image_index = SQL_indexer(query_string, connection)
        self.resize = resize
        self.upright = True

    # @lru_cache(maxsize=None)
    def __call__(self, row, key=None):  # redefined to (192.256)
        rec = row
        if key is None:
            img_index = row["index"]
        else:
            img_index = key
        img = pickle.loads(self.image_index[img_index])  # strange autoincrement feature
        # new_y2 = int((rec.y2-rec.y1)/3+rec.y1)#1/3 head patch
        result = img[rec.y1 : rec.y2, rec.x1 : rec.x2]
        # result = img[rec.y1:new_y2,rec.x1:rec.x2]
        result = fill_hole(result)
        p1 = np.percentile(result.flatten(), 1)
        p99 = np.percentile(result.flatten(), 99)
        result = np.clip(result, p1, p99)
        if self.upright == True:
            result = np.rot90(result)
        if self.resize is not None:
            result = cv2.resize(result, self.resize)
        return result
