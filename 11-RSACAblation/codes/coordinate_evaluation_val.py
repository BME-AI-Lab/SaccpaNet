# %%
import gc
import importlib
import os
from glob import glob

import pytorch_lightning as pl
import torch
from search_configs import MODEL_NAME, params, uattention_arms, uattention_atrous

# from configs.manually_searched_params import params
from lib.procedures import load_pretrained_kpt, update_log, write_log
from lib.procedures.dataloaders_procedure import (
    create_test_dataloader,
    create_validation_dataloader,
)

if __name__ == "__main__":
    # MODEL_NAME = "SACCPA_sample"
    EVALUATION_MODEL = glob(f"./log/{MODEL_NAME}/**/last*.ckpt", recursive=True)[0]
    model = importlib.import_module(f"models.{MODEL_NAME}")
    BATCH_SIZE = 1
    default_root_dir = f"./log/{MODEL_NAME}"
    test_dataloader = create_validation_dataloader(BATCH_SIZE, WITH_QUILT=True)
    PARAM_NAME = os.path.basename(os.getcwd())
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    kpt_model = model.MyLightningModule(
        params,
        num_joints=18,
        uattention_arms=uattention_arms,
        uattention_atrous=uattention_atrous,
    )
    print(EVALUATION_MODEL)
    state_dict = torch.load(EVALUATION_MODEL)
    kpt_model.load_state_dict(state_dict["state_dict"])

    # kpt_model = load_pretrained_kpt(MODEL_NAME, EVALUATION_MODEL, params)

    kpt_model.eval()
    # model = model.MyLightningModule(kpt_model, params)
    trainer = pl.Trainer(gpus=[0], amp_level="O2", accelerator="dp", amp_backend="apex")
    x = trainer.test(kpt_model, test_dataloader)
    print(MODEL_NAME, x)
