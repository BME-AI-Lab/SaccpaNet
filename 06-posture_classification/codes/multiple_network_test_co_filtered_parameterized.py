# %%
import os
from numpy import ModuleDeprecationWarning
import torch
import torchvision.models
import importlib
import csv

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
from lib.modules.core.sampler import sample_cfgs, generate_regnet_full

# %%
import pytorch_lightning as pl

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

    MODEL_NAME = "segnext_sample"
    seed = 6
    model = importlib.import_module(f"models.{MODEL_NAME}")
    SAMPLES = sample_cfgs(seed=seed, model=model.MyLightningModule)
    BATCH_SIZE = 32
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
    for PARAM_NAME, params in SAMPLES.items():
        # MODEL_NAME = "modelD"
        model = importlib.import_module(f"models.{MODEL_NAME}")
        model = model.MyLightningModule(params)  # locals()[f'{MODEL_NAME}']()
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f"Now training for {MODEL_NAME}")
        trainer = pl.Trainer(
            gpus=[0],
            amp_level="O2",
            accelerator="dp",
            amp_backend="apex",
            max_epochs=100,
            min_epochs=100,
            default_root_dir=default_root_dir,
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
        params["seed"] = seed
        params["params"] = pytorch_total_params
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
            # f.write(str(params))
            # f.write("\n\n\n")
        with open(log_file + ".txt", "a") as f:
            f.write(str(pl.utilities.model_summary.summarize(model)))

        del trainer
        del model
