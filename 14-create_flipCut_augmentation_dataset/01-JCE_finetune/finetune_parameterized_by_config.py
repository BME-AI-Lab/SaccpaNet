# %%
import gc
import importlib
import os

import torch
from augmented_dataloader_procedures import create_dataloaders

from configs.manually_searched_params import params
from lib.procedures import train_and_evaluate, update_log, write_log

if __name__ == "__main__":
    MODEL_NAME = "SACCPA_sample"
    PRETRAIN_MODEL = "./best-epoch.ckpt"
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
    state_dict = torch.load(PRETRAIN_MODEL)
    model.load_state_dict(state_dict["state_dict"])
    epoch = 150
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
