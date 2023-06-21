from lib.procedures import (
    create_load_cls_kpt,
    create_validation_dataloader,
    inference_model_classification_coordinate,
    evaluate_cls,
)
from sklearn.metrics import *
from configs.manually_searched_params import params

KEYPOINT_MODELS = "saccpa_sample"
CLASSIFICATION_MODELS = "ScappaClass"
ckpt_path = "log\\saccpa_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"

BATCH_SIZE = 1
default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
NO_QUILT_TRAIN = False
MIX_TRAIN = True
WITH_QUILT = True
VALIDATION = True
ALL_CONDITIONS_STRING = (
    f"TrainQuilt{NO_QUILT_TRAIN}_MixTrain{MIX_TRAIN}_TestWithQuilt{WITH_QUILT}"
)

test_dataloader = create_validation_dataloader(BATCH_SIZE, WITH_QUILT, VALIDATION)
model, RESULT_DIR = create_load_cls_kpt(
    KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path, params, default_root_dir
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
