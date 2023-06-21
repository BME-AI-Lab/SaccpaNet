import csv
import gc
from glob import glob
import os
from os.path import dirname
import pytorch_lightning as pl
import importlib


import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.manually_searched_params import params
from configs.random_searched_params import params
from .dataloaders_procedure import create_dataloaders


def load_train_eval_sample(
    MODEL_NAME,
    model,
    default_root_dir,
    train_dataloader,
    test_dataloader,
    epochs,
):
    print(f"Now training for {MODEL_NAME}")
    trainer = train_model(
        model, default_root_dir, train_dataloader, test_dataloader, epochs
    )
    x = trainer.test(model, test_dataloader, verbose=True)
    return model, trainer, x


def create_kpt_model(KEYPOINT_MODELS, params):
    kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
    kpt_model = kpt_model.MyLightningModule(params, num_classes=18)

    return kpt_model


def create_classification_with_keypoint_model(
    KEYPOINT_MODELS,
    CLASSIFICATION_MODELS,
    ckpt_path,
):
    kpt_model = load_pretrained_keypoint_model(KEYPOINT_MODELS, ckpt_path)
    model = create_cls_model(CLASSIFICATION_MODELS, kpt_model)
    return model


def create_cls_model(CLASSIFICATION_MODELS, kpt_model):
    model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
    model = model.MyLightningModule(kpt_model)
    return model


def load_pretrained_keypoint_model(KEYPOINT_MODELS, ckpt_path, params):
    kpt_model = create_kpt_model(KEYPOINT_MODELS, params)
    kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
    kpt_model.eval()
    return kpt_model


def create_model_coordinate_classification(
    KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path, params, default_root_dir
):
    kpt_model = load_pretrained_keypoint_model(KEYPOINT_MODELS, ckpt_path, params)
    model = create_cls_model(CLASSIFICATION_MODELS, kpt_model)
    print(default_root_dir)
    # find_files = glob(f"{default_root_dir}/lightning_logs/*/checkpoints/*.ckpt")
    model, RESULT_DIR = load_cls_model(default_root_dir, model)
    return model, RESULT_DIR


def load_cls_model(default_root_dir, model):
    find_files = glob(f"{default_root_dir}/lightning_logs/*/checkpoints/last*.ckpt")
    assert len(find_files) > 0
    find_files.sort()
    model_checkpoint_file = find_files[-1]
    load = torch.load(model_checkpoint_file)
    model.load_state_dict(load["state_dict"])
    RESULT_DIR = dirname(model_checkpoint_file)
    model.cuda()
    model.eval()
    return model, RESULT_DIR


def train_model(
    model,
    default_root_dir,
    train_dataloader,
    test_dataloader,
    epoch,
):
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        # dirpath=default_root_dir,
        filename="best-{epoch:03d}-{val_loss:.3f}",
    )
    checkpoint_callback_last = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        # dirpath=default_root_dir,
        filename="last-{epoch:03d}-{global_step}",
    )
    trainer = pl.Trainer(
        gpus=[0],
        amp_level="O2",
        accelerator="dp",
        amp_backend="apex",
        max_epochs=epoch,
        min_epochs=epoch,
        default_root_dir=default_root_dir,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last],
    )  # gpus=1, accelerator='dp',
    trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)
    return trainer
