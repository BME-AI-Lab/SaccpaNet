# %%
import torch
import torchvision.models
import importlib

from lib.procedures import create_mode_coordinate_classification
from lib.procedures import create_dataloader_coordinate_evaluation
from lib.procedures import inference_model_classification_coordinate
from lib.procedures import evaluate_classification_model

importlib.reload(torchvision)
torch.__version__

from sklearn.metrics import *
from configs.manually_searched_params import params


# MODELS = ["modelD","modelA","modelC","modelH","modelE","test1","modelB","model0","modelF","test4_1","test4_2","test4_3","test4_5"]
KEYPOINT_MODELS = "segnext_sample"
CLASSIFICATION_MODELS = "ScappaClass"
ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"

if True:
    BATCH_SIZE = 1
    # MODEL_NAME =#"modelD"
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"  # without_mix/
    # train_dataset = SQLJointsDataset(train=True)
    NO_QUILT_TRAIN = False
    MIX_TRAIN = True
    WITH_QUILT = True
    VALIDATION = True
    # assert not(MIX_TRAIN == False and NO_QUILT_TRAIN=True)

    test_dataloader = create_dataloader_coordinate_evaluation(
        BATCH_SIZE, WITH_QUILT, VALIDATION
    )
    ALL_CONDITIONS_STRING = (
        f"TrainQuilt{NO_QUILT_TRAIN}_MixTrain{MIX_TRAIN}_TestWithQuilt{WITH_QUILT}"
    )

    model, RESULT_DIR = create_mode_coordinate_classification(
        KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path, params, default_root_dir
    )
    (
        ly,
        ly_hat,
        image_ids,
        ly_weight,
        input_storage,
    ) = inference_model_classification_coordinate(test_dataloader, model)

    evaluate_classification_model(
        ALL_CONDITIONS_STRING,
        RESULT_DIR,
        ly,
        ly_hat,
        image_ids,
        ly_weight,
        input_storage,
    )
