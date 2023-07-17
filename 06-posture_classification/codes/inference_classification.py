from get_classification_model import classification_models
from search_configs import CLASSIFICATION_MODELS
from sklearn.metrics import *

from configs.manually_searched_params import params
from lib.procedures import (
    create_test_dataloader,
    create_validation_dataloader,
    evaluate_cls,
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
cls_model = classification_models[CLASSIFICATION_MODELS](num_classes=1000)
for VALIDATION in [True, False]:
    ALL_CONDITIONS_STRING = f"TestWithQuilt{WITH_QUILT}_Validation{VALIDATION}"
    if VALIDATION:
        test_dataloader = create_validation_dataloader(
            BATCH_SIZE, WITH_QUILT, VALIDATION
        )
    else:
        test_dataloader = create_test_dataloader(BATCH_SIZE)
    model, RESULT_DIR = load_cls_kpt(
        KEYPOINT_MODELS, cls_model, ckpt_path, params, default_root_dir
    )
    (
        ly,
        ly_hat,
        image_ids,
        ly_weight,
        input_storage,
    ) = inference_model_classification_coordinate(test_dataloader, model)

    evaluate_cls(
        ALL_CONDITIONS_STRING,
        RESULT_DIR,
        ly,
        ly_hat,
        image_ids,
        ly_weight,
        input_storage,
    )
