import importlib

import torch

from configs.manually_searched_params import params

if __name__ == "__main__":
    MODEL_NAME = "SACCPA_sample"
    PRETRAIN_MODEL = "template_checkpoint.pth"
    kpt_model = importlib.import_module(f"models.{MODEL_NAME}")
    kpt_model = kpt_model.MyLightningModule(params, num_joints=18)
    torch.save(kpt_model.state_dict(), PRETRAIN_MODEL)
