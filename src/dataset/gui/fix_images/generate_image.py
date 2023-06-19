from common import SQL_indexer
import pickle
import cv2
import uuid
import numpy as np
import sqlalchemy
from functools import lru_cache


class ImageHandler:
    def initialize(self, database, connection):
        self.database = database
        self.annotation_database = "data_06_15_annotations"
        self.connection = connection
        self.query_string = (
            "SELECT a.depth_array FROM " + self.database + " as a WHERE a.index={}"
        )

        # openpose settings
        self.num_joints = 18
        self.indexs = self._get_sql_indexs()
        self.joint_database = (
            "openpose_annotation_06_02"  # "openpose_annotation_03_18_with_quilt" #
        )
        self.sigma = 2

    @lru_cache
    def get_image(self, id):
        img_index = index = id
        img = pickle.loads(self.image_index[img_index])
        self.conn = sqlalchemy.create_engine(self.connection, echo=False)
        query_string = (
            "SELECT a.x1,x2,y1,y2 FROM "
            + self.annotation_database
            + " as a WHERE a.index={}"
        )
        result = self.conn.execute(query_string.format(index))
        print(query_string.format(index))
        assert result.rowcount == 1
        x1, x2, y1, y2 = result.first()
        img = img[y1:y2, x1:x2]
        return img

    def img_to_vis(self, image):
        raise NotImplementedError

    def overlay_openpose(self, image, index):
        joints_3d, joints_vis = self._get_joints(index)

        # print(joints_3d,joints_vis)
        target, target_weight = self._generate_joint_heatmap(
            image, joints_3d, joints_vis
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = image.copy()
        for i in range(18):
            xy = (int(image.shape[1] - joints_3d[i, 1]), int(joints_3d[i, 0]))
            image = cv2.putText(image, str(i), xy, font, fontScale, color, thickness)

        return self._patch_image(image, target)

    def _patch_image(self, image, target):
        # target = np.rot90(target)
        target = np.sum(target, axis=0)
        target = np.rot90(target, k=3)
        # print(image.shape,target.shape)
        threshold = 0.5
        if len(image.shape) != 3:  # rgb
            image[target > threshold] = 255
        else:
            image[:, :, 0][target > threshold] = 255
        return image

    def _get_joints(self, index):
        self.conn = sqlalchemy.create_engine(self.connection, echo=False)
        self.indexs = self._get_sql_indexs()

        query_string = (
            f"SELECT {self.indexs} FROM {self.joint_database}"
            + " as a WHERE a.image_id={}"
        )
        result = self.conn.execute(query_string.format(index))
        # print(query_string.format(index))
        assert result.rowcount == 1

        result = list(result.first())
        print(result)
        joints = np.array(result[1:])
        joints = self.patch2(joints)  # for interactive movement
        # remark xs,ys swaped due to upright
        xs, ys = (
            joints[0::2],
            joints[1::2],
        )  # TODO: check if it is the case OR place assertion
        _vis = np.int32((xs != 0.0) & (ys != 0.0))
        zeros = np.zeros_like(xs)
        joints_vis = np.stack([_vis, _vis, zeros], axis=1)
        joints_3d = np.stack([xs, ys, zeros], axis=1)
        del self.conn
        return joints_3d, joints_vis

    def _get_sql_indexs(self):
        num_index = self.num_joints
        indexs = ["a.index"]
        for i in range(num_index):
            indexs.append(f"a.{i}_x,a.{i}_y")
        # print(indexs)
        return ",".join(indexs)

    def _generate_joint_heatmap(self, image, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        self.heatmap_size = np.array(image.shape[:2])
        self.image_size = self.heatmap_size
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        # print(target_weight,joints_vis)
        target_weight[:, 0] = joints_vis[:, 0]
        self.target_type = "gaussian"
        if self.target_type == "gaussian":
            target = np.zeros(
                (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if (
                    ul[0] >= self.heatmap_size[0]
                    or ul[1] >= self.heatmap_size[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                        g_y[0] : g_y[1], g_x[0] : g_x[1]
                    ]
        return target, target_weight

    def get(self, img_id):
        self.image_index = SQL_indexer(self.query_string, self.connection)
        overlay = True  # self.get_argument("overlay","false") != "false"
        img_id = int(img_id)
        img = self.get_image(img_id)
        del self.image_index
        if overlay:
            img = self.overlay_openpose(img, img_id)
        img = self.img_to_vis(img)
        # tmp_filename = f"tmp/{uuid.uuid1()}.png"
        # cv2.imwrite(tmp_filename,img)
        # self.redirect(f"/{tmp_filename}",permanent=True)
        return img

    def patch(self, joints_3d):
        for joint in range(18):
            if joint in self.patch_list:
                joints_3d[joint, 0] = self.patch_list[joint][0]
                joints_3d[joint, 1] = self.patch_list[joint][1]
        return joints_3d

    def patch2(self, joints_3d):
        for joint in range(18):
            if joint in self.patch_list:
                joints_3d[joint * 2 + 0] = self.patch_list[joint][0]
                joints_3d[joint * 2 + 1] = self.patch_list[joint][1]
        return joints_3d

    def set_patch(self, joint, x, y):
        self.patch_list[joint] = [x, y]

    def commit_patch(self, index):
        columns = []
        for joint in self.patch_list:
            x, y = self.patch_list[joint]
            columns += [f"{joint}_x={x}, {joint}_y={y}"]
        if len(columns) > 0:
            columns = ",".join(columns)
            query_string = (
                f"UPDATE {self.joint_database} SET {columns} WHERE image_id={index}"
            )
            self.conn = sqlalchemy.create_engine(self.connection, echo=False)
            result = self.conn.execute(query_string)
            self.patch_list = {}
            print("commited")
            pass


class DepthImageHandler(ImageHandler):
    """Get Raw Depth image without crop

    Args:
        tornado (_type_): _description_
    """

    def initialize(self, database, connection):
        super().initialize(database, connection)
        self.database = database
        self.connection = connection
        self.query_string = (
            "SELECT a.depth_array FROM " + self.database + " as a WHERE a.index={}"
        )
        self.patch_list = {}
        self.image_index = SQL_indexer(self.query_string, self.connection)

    def img_to_vis(self, image):
        p1 = np.percentile(image.flatten(), 5)
        p99 = np.percentile(image.flatten(), 95)
        # image = (image-p1)/(p99-p1) *255
        # image = image.astype(np.uint8)
        image = cv2.convertScaleAbs(image, alpha=255 / (p99 - p1), beta=-p1)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        image = np.rot90(image)
        # image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return image


class RGBImageHandler(ImageHandler):
    """Get Raw Depth image without crop

    Args:
        tornado (_type_): _description_
    """

    def initialize(self, database, connection):
        super().initialize(database, connection)
        self.database = database
        self.connection = connection
        self.query_string = (
            "SELECT a.color_frame_image_align FROM "
            + self.database
            + " as a WHERE a.index={}"
        )
        self.image_index = SQL_indexer(self.query_string, self.connection)
        self.patch_list = {}

    def img_to_vis(self, image):
        image = image[:, :, (2, 1, 0)]  # bgr to rgb
        image = np.rot90(image)
        return image
