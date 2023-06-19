# %%
from numpy import ModuleDeprecationWarning
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
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.modules.modules import Network

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# from models.modelA import MyLightningModule as modelA

# from models.modelB import MyLightningModule as modelB
# from models.modelD import MyLightningModule as modelD
# from models.modelH import MyLightningModule as modelH

# from models.modelE import MyLightningModule as modelE
if __name__ == "__main__":
    import importlib

    KEYPOINT_MODELS = "segnext_sample"
    CLASSIFICATION_MODELS = "EfficientNetCoord"
    ckpt_path = "log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_6\\checkpoints\\best-epoch=069-val_loss=0.344.ckpt"
    params = {
        "REGNET.DEPTH": 28,
        "REGNET.W0": 104,
        "REGNET.WA": 35.7,
        "REGNET.WM": 2,
        "REGNET.GROUP_W": 40,
        "REGNET.BOT_MUL": 1,
    }
    if True:
        BATCH_SIZE = 32
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
                num_workers=8,
                pin_memory=True,
                persistent_workers=8,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=8,
            )
            kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
            kpt_model = kpt_model.MyLightningModule(
                params, num_classes=18
            )  # locals()[f'{MODEL_NAME}']()
            kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
            kpt_model.eval()
            model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
            model = model.MyLightningModule(kpt_model)
            # checkpoints
            checkpoint_callback_best = ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                # dirpath=default_root_dir,
                filename="best-{epoch:03d}-{val_loss:.3f}",
            )
            checkpoint_callback_last = ModelCheckpoint(
                save_top_k=1,
                monitor="epoch",
                mode="max",
                # dirpath=default_root_dir,
                filename="last-{epoch:03d}-{global_step}",
            )
            # training
            print(f"Now training for {CLASSIFICATION_MODELS}+{KEYPOINT_MODELS}")
            trainer = pl.Trainer(
                gpus=[0],
                amp_level="O2",
                accelerator="dp",
                amp_backend="apex",
                max_epochs=500,
                min_epochs=500,
                default_root_dir=default_root_dir,
                callbacks=[checkpoint_callback_best, checkpoint_callback_last],
                # resume_from_checkpoint="log\\ScappaClass\\lightning_logs\\version_0\\checkpoints\\epoch=46-step=3666.ckpt",
            )  # gpus=1, accelerator='dp',
            trainer.tune(model, train_dataloader)
            trainer.fit(model, train_dataloader, test_dataloader)

            # %%
            # (model.net._avg_pooling.output_size)
            # %%
            x = trainer.test(model, test_dataloader, verbose=True)

            # %%
            print(x)
