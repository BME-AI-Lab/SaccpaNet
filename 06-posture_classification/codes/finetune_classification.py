# %%
import torch
import torchvision.models
import importlib

from lib.procedures import create_classification_with_keypoint_model
from lib.procedures import train_for_model_finetune_classification

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset

# %%

# import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from configs.manually_searched_params import params

from lib.procedures import *


if __name__ == "__main__":
    import importlib

    KEYPOINT_MODELS = "segnext_sample"
    CLASSIFICATION_MODELS = "ScappaClass"
    ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
    BATCH_SIZE = 16
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    if True:
        # MODEL_NAME = "modelD"

        train_dataloader, test_dataloader = create_dataloaders(BATCH_SIZE)

        model = create_classification_with_keypoint_model(
            KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path
        )
        epoch = 500
        MODEL_NAME = f"{CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}"
        model, trainer, x = load_train_eval_sample(
            MODEL_NAME,
            model,
            default_root_dir,
            train_dataloader,
            test_dataloader,
            epoch=epoch,
        )

        # %%
        # (model.net._avg_pooling.output_size)
        # %%
        x = trainer.test(model, test_dataloader, verbose=True)

        # %%
        print(x)
