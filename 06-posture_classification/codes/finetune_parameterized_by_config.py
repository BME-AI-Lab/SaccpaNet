# %%
import os
import torch
import torchvision.models
import importlib
import csv
from configs.manually_searched_params import params

importlib.reload(torchvision)
torch.__version__
# from config import *


# %%
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
import gc

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# from models.modelA import MyLightningModule as modelA

# from models.modelB import MyLightningModule as modelB
# from models.modelD import MyLightningModule as modelD
# from models.modelH import MyLightningModule as modelH

# from models.modelE import MyLightningModule as modelE
if __name__ == "__main__":
    import importlib

    MODEL_NAME = "segnext_sample"
    PRETRAIN_MODEL = "merged_model.pth"
    model = importlib.import_module(f"models.{MODEL_NAME}")
    BATCH_SIZE = 8
    if True:  # dataset
        test_dataset = SQLJointsDataset(train=False)

        WITH_FILTERED = True
        if MODEL_NAME == "model0":
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
        # for WITHOUT_MIX in [False]: #temp script
        if WITHOUT_MIX:
            train_dataset.probability = 0
            default_root_dir = f"./log/without_mix/{MODEL_NAME}"
        else:
            default_root_dir = f"./log/{MODEL_NAME}"
        # sampler specific
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
    old_default_root_dir = default_root_dir
    # for PARAM_NAME,params in SAMPLES.items():
    if True:
        PARAM_NAME = os.path.basename(os.getcwd())
        # MODEL_NAME = "modelD"
        default_root_dir = f"{old_default_root_dir}/{PARAM_NAME}"
        model = importlib.import_module(f"models.{MODEL_NAME}")
        model = model.MyLightningModule(
            params, num_classes=18
        )  # locals()[f'{MODEL_NAME}']()

        state_dict = torch.load(PRETRAIN_MODEL)
        model.load_state_dict(state_dict)
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
        # trainig
        print(f"Now training for {MODEL_NAME}")
        trainer = pl.Trainer(
            gpus=[0],
            amp_level="O2",
            accelerator="dp",
            amp_backend="apex",
            max_epochs=100,
            min_epochs=100,
            default_root_dir=default_root_dir,
            callbacks=[checkpoint_callback_best, checkpoint_callback_last],
            resume_from_checkpoint="log\\segnext_sample\\15-fine_tuning\\lightning_logs\\version_5\\checkpoints\\best-epoch=051-val_loss=0.355.ckpt",
        )  # gpus=1, accelerator='dp',
        trainer.tune(model, train_dataloader)
        trainer.fit(model, train_dataloader, test_dataloader)

        # %%
        # (model.net._avg_pooling.output_size)
        # %%
        x = trainer.test(model, test_dataloader, verbose=True)

        # %%
        params.update(x[0])
        params["key"] = PARAM_NAME
        log_file = "log.csv"

        if os.path.isfile(log_file):
            write_header = False
        else:
            write_header = True

        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, params.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(params)

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
