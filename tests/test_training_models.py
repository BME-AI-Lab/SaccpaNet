# %%
import gc
import importlib
import os

import pytorch_lightning as pl
import torch
from torchview import draw_graph
from torchvision.models import resnet18

from configs.manually_searched_params import params

# import helper.tqdm_hook  # progress bar report  hack
from configs.random_searched_params import params
from lib.procedures import *
from lib.procedures import (
    create_cls,
    create_cls_kpt,
    create_dataloaders,
    create_kpt,
    load_pretrained_kpt,
    train_and_evaluate,
    update_log,
    write_log,
)


def test_saccpa_sample():
    from models.SACCPA_sample import MyLightningModule

    MODEL_NAME = "SACCPA_sample"
    model = MyLightningModule(params)
    BATCH_SIZE = 8
    default_root_dir = f"./log/{MODEL_NAME}"
    (
        train_dataloader,
        val_dataloader,
    ) = create_dataloaders(BATCH_SIZE)
    PARAM_NAME = os.path.basename(os.getcwd())
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    epoch = 1
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epoch,
        min_epochs=epoch,
        default_root_dir=default_root_dir,
        limit_train_batches=1 / 168,
        limit_test_batches=0.05,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, val_dataloader, verbose=True)


def test_classification_network():
    from models.ClassificationWithCoordinate import MyLightningModule

    KEYPOINT_MODELS = "SACCPA_sample"
    CLASSIFICATION_MODELS = "ClassificationWithCoordinate"
    BATCH_SIZE = 16
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    (
        train_dataloader,
        test_dataloader,
    ) = create_dataloaders(BATCH_SIZE)

    kpt_model = create_kpt(KEYPOINT_MODELS, params)
    cls_model = resnet18(pretrained=True)
    model = MyLightningModule(kpt_model, cls_model)
    MODEL_NAME = f"{CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}"
    epoch = 1
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epoch,
        min_epochs=epoch,
        default_root_dir=default_root_dir,
        limit_train_batches=1 / 168,
        limit_test_batches=0.05,
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader, verbose=True)
