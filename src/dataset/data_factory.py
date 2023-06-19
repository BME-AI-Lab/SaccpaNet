

class DataFactory:
    
    @classmethod
    def get_row_by_image_id(id):
        pass
    def get_image

class NotModifiableError(Exception):
    """Raise for attempt to modify a record during training"""

class DataRow:
    # unique_identifier is image id 
    EDITABLE = False
    
    def _get_sql(self):
    
    @property
    def joints(self):
        pass
    
    @joints.setter
    def joints(self, value):
        if self.EDITABLE:
            self._joints = value
        else:
            raise NotModifiableError("Cannot modify joints")
    
    @property
    def depth_image(self):
        pass
    
    @property
    def color_image(self):
        pass

    def update_database():
        pass
    
    @property
    def joint_annotation_image(self):
        
    @property
    def joint_annotaion_heatmap(self):
        self.heatmap_size = np.array(image.shape[:2])
        self.image_size = self.heatmap_size
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        print(target_weight,joints_vis)
        target_weight[:, 0] = joints_vis[:, 0]
        self.target_type = 'gaussian'
        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight 