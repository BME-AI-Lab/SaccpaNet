import importlib

import torch

from configs.manually_searched_params import params

if __name__ == "__main__":
    MODEL_NAME = "saccpa_sample"
    PRETRAIN_MODEL = "template_checkpoint.pth"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    torch.save(model.state_dict(), PRETRAIN_MODEL)
