# %%
import gc
import importlib
import os

import torch
from search_configs import MODEL_NAME

from configs.manually_searched_params import params
from lib.procedures import create_dataloaders, train_and_evaluate, update_log, write_log

if __name__ == "__main__":
    model = importlib.import_module(f"models.{MODEL_NAME}")
    BATCH_SIZE = 8
    default_root_dir = f"./log/{MODEL_NAME}"
    (
        train_dataloader,
        val_dataloader,
    ) = create_dataloaders(BATCH_SIZE)
    PARAM_NAME = os.path.basename(os.getcwd())
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    model = model.MyLightningModule(params, num_joints=18)
    epoch = 600
    model, trainer, x = train_and_evaluate(
        MODEL_NAME,
        model,
        default_root_dir,
        train_dataloader,
        val_dataloader,
        epoch,
    )
    update_log(PARAM_NAME, params, x)

    log_file = "log.csv"
    write_log(params, log_file)
