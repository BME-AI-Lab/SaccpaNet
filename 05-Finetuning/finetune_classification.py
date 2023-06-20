# %%
import torch
import torchvision.models
import importlib

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset

# %%
import pytorch_lightning as pl

# import torchvision.models as models
from torch.utils.data.dataloader import DataLoader

from configs.manually_searched_params import params


if __name__ == "__main__":
    import importlib

    KEYPOINT_MODELS = "segnext_sample"
    CLASSIFICATION_MODELS = "ScappaClass"
    ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
    BATCH_SIZE = 32
    if True:
        # MODEL_NAME = "modelD"

        test_dataset = SQLJointsDataset(train=False)
        WITH_FILTERED = True
        if CLASSIFICATION_MODELS == "model0":
            # test_dataset = SQLJointsDataset(train=False)
            SQLJointsDataset.TABLE_NAME = "openpose_annotation_old_data"
            train_dataset = SQLJointsDataset(train=True)
        elif WITH_FILTERED == True:
            SQLJointsDataset.TABLE_NAME = (
                "openpose_annotation_03_18_with_quilt_filtered"
            )
            train_dataset = SQLJointsDataset(train=True)
        else:
            train_dataset = SQLJointsDataset(train=True)
        WITHOUT_MIX = False
        for WITHOUT_MIX in [False]:  # temp script
            if WITHOUT_MIX:
                train_dataset.probability = 0
                default_root_dir = f"./log/without_mix/{CLASSIFICATION_MODELS}"
            else:
                default_root_dir = f"./log/{CLASSIFICATION_MODELS}"
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                persistent_workers=4,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                persistent_workers=4,
            )
            kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
            kpt_model = kpt_model.MyLightningModule(
                params, num_classes=18
            )  # locals()[f'{MODEL_NAME}']()
            kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
            kpt_model.eval()
            model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
            model = model.MyLightningModule(kpt_model)

            print(f"Now training for {CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}")
            trainer = pl.Trainer(
                gpus=[0],
                amp_level="O2",
                accelerator="dp",
                amp_backend="apex",
                max_epochs=500,
                min_epochs=500,
                default_root_dir=default_root_dir,
            )  # gpus=1, accelerator='dp',
            trainer.tune(model, train_dataloader)
            trainer.fit(model, train_dataloader, test_dataloader)

            # %%
            # (model.net._avg_pooling.output_size)
            # %%
            x = trainer.test(model, test_dataloader, verbose=True)

            # %%
            print(x)
