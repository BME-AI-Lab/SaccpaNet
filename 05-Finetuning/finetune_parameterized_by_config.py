# %%
import gc
import importlib
import os

import torch

from configs.manually_searched_params import params
from lib.procedures import create_dataloaders, train_and_evaluate, update_log, write_log

if __name__ == "__main__":
    MODEL_NAME = "saccpa_sample"
    PRETRAIN_MODEL = "merged_model.pth"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    BATCH_SIZE = 8
    default_root_dir = f"./log/{MODEL_NAME}"
    (
        train_dataloader,
        test_dataloader,
    ) = create_dataloaders(BATCH_SIZE)
    PARAM_NAME = os.path.basename(os.getcwd())
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    model = model.MyLightningModule(params, num_classes=18)
    state_dict = torch.load(PRETRAIN_MODEL)
    model.load_state_dict(state_dict)
    epoch = 300
    model, trainer, x = train_and_evaluate(
        MODEL_NAME,
        model,
        default_root_dir,
        train_dataloader,
        test_dataloader,
        epoch,
    )
    update_log(PARAM_NAME, params, x)

    log_file = "log.csv"
    write_log(params, log_file)
