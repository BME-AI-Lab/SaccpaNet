import os

import numpy as np

DATALOADER_WORKERS = 4
SQLITE_DB_PATH = "C:\\Posture Coordinate Models\\dataset.db"
DB_CONNECTION_STRING = f"sqlite:///{SQLITE_DB_PATH}"

# Augmentation dataset setup
FLIPCUT_AUGMENTED_SQLITE_DB_PATH = "C:\\Posture Coordinate Models\\augmented_dataset.db"
FLIPCUT_AUGMENTED_DB_CONNECTION_STRING = f"sqlite:///{FLIPCUT_AUGMENTED_SQLITE_DB_PATH}"

# IMAGE SETUP
IMAGE_SIZE = np.array((192, 256))
HEATMAP_SIZE = np.array((48, 64))
TARGET_TYPE = "gaussian"
SIGMA = 2

# Transformations
SCALE_FACTOR = 0.2  # 1+/- this factor
ROTATION_FACTOR = 15  # Degree   # TBD: double check if the unit is degree


if os.environ.get("SQLITE_DB_PATH"):
    SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH")
    DB_CONNECTION_STRING = f"sqlite:///{SQLITE_DB_PATH}"

from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
