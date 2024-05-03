import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


from get_classification_model import classification_models
from inference_lib import *
from joblib import Memory
from search_configs import CLASSIFICATION_MODELS
from sklearn.metrics import *

from configs.manually_searched_params import params
from lib.procedures import (
    create_test_dataloader,
    create_validation_dataloader,
    inference_model_classification_coordinate,
    load_cls_kpt,
)

KEYPOINT_MODELS = "SACCPA_sample"
ckpt_path = "../best-epcoh.ckpt"

BATCH_SIZE = 1
default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
NO_QUILT_TRAIN = False
MIX_TRAIN = True
WITH_QUILT = True
VALIDATION = True


def calculate_flops(test_dataloader, model):
    from thop import profile

    input = next(iter(test_dataloader))[0].cuda()
    macs, params = profile(model, inputs=(input,))
    # print(macs, params)
    return macs, params


if __name__ == "__main__":
    cls_model = classification_models[CLASSIFICATION_MODELS](num_classes=1000)
    for VALIDATION in [False]:
        ALL_CONDITIONS_STRING = f"TestWithQuilt{WITH_QUILT}_Validation{VALIDATION}"
        if VALIDATION:
            test_dataloader = create_validation_dataloader(BATCH_SIZE, WITH_QUILT)
        else:
            test_dataloader = create_test_dataloader(BATCH_SIZE)
        model, RESULT_DIR = load_cls_kpt(
            KEYPOINT_MODELS, cls_model, ckpt_path, params, default_root_dir
        )
        model.eval()
        with torch.no_grad():
            # memory = Memory("./tmp", verbose=0)
            # inference_clasification_model = memory.cache(inference_clasification_model)
            # df = inference_clasification_model(test_dataloader, model)
            macs, params = calculate_flops(test_dataloader, model)
            print(f"model:{CLASSIFICATION_MODELS} macs: {macs}, params: {params}")
