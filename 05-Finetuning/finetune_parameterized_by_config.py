# %%
import os
import torch
import torchvision.models
import importlib
from configs.random_searched_params import params
from lib.procedures import train_and_evaluate_finetune_parameterized_by_config
from lib.procedures import create_dataloaders

importlib.reload(torchvision)
torch.__version__
# from config import *

# grid searched

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%

# %%

# import torchvision.models as models

if __name__ == "__main__":
    import importlib

    MODEL_NAME = "segnext_sample"
    PRETRAIN_MODEL = "merged_model.pth"
    seed = 6
    model = importlib.import_module(f"models.{MODEL_NAME}")
    BATCH_SIZE = 8
    default_root_dir = f"./log/{MODEL_NAME}"
    (
        train_dataloader,
        test_dataloader,
    ) = create_dataloaders(MODEL_NAME, BATCH_SIZE)
    PARAM_NAME = os.path.basename(os.getcwd())
    default_root_dir = f"{default_root_dir}/{PARAM_NAME}"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    model = model.MyLightningModule(params, num_classes=18)

    train_and_evaluate_finetune_parameterized_by_config(
        MODEL_NAME,
        PRETRAIN_MODEL,
        model,
        default_root_dir,
        train_dataloader,
        test_dataloader,
        PARAM_NAME,
    )
