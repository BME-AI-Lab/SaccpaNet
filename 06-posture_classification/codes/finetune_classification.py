# %%
import torch
import torchvision.models
import importlib

from lib.procedures import create_model_classification_with_keypoint
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


from lib.procedures import create_dataloader


if __name__ == "__main__":
    import importlib

    KEYPOINT_MODELS = "segnext_sample"
    CLASSIFICATION_MODELS = "ScappaClass"
    ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
    BATCH_SIZE = 16
    default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
    if True:
        # MODEL_NAME = "modelD"

        train_dataloader, test_dataloader = create_dataloader(
            CLASSIFICATION_MODELS, BATCH_SIZE
        )

        model = create_model_classification_with_keypoint(
            KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path
        )

        print(f"Now training for {CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}")
        trainer = train_for_model_finetune_classification(
            default_root_dir, train_dataloader, test_dataloader, model
        )

        # %%
        # (model.net._avg_pooling.output_size)
        # %%
        x = trainer.test(model, test_dataloader, verbose=True)

        # %%
        print(x)
