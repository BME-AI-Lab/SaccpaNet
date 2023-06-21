import csv
import gc
import os
import pytorch_lightning as pl
import importlib

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from configs.manually_searched_params import params
from configs.random_searched_params import params
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset


from torch.utils.data.dataloader import DataLoader


def create_dataloader(MODEL_NAME, BATCH_SIZE):
    test_dataset = SQLJointsDataset(train=False)
    train_dataset = SQLJointsDataset(train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=2,
    )

    return train_dataloader, test_dataloader


def load_train_eval_sample(
    MODEL_NAME, default_root_dir, train_dataloader, test_dataloader, params, epochs
):
    model = create_model_by_params(MODEL_NAME, params)  # locals()[f'{MODEL_NAME}']()

    print(f"Now training for {MODEL_NAME}")
    trainer = pl.Trainer(
        gpus=[0],
        amp_level="O2",
        accelerator="dp",
        amp_backend="apex",
        max_epochs=epochs,
        min_epochs=epochs,
        default_root_dir=default_root_dir,
    )  # gpus=1, accelerator='dp',
    trainer.tune(model, train_dataloader)
    trainer.fit(model, train_dataloader, test_dataloader)

    # %%
    # (model.net._avg_pooling.output_size)
    # %%
    x = trainer.test(model, test_dataloader, verbose=True)
    return model, trainer, x


def create_model_by_params(MODEL_NAME, params):
    model = importlib.import_module(f"models.{MODEL_NAME}")
    model = model.MyLightningModule(params)
    return model


def write_log(params, log_file):
    if os.path.isfile(log_file):
        write_header = False
    else:
        write_header = True

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, params.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(params)


def update_log(PARAM_NAME, params, x):
    params.update(x[0])
    params["key"] = PARAM_NAME


def create_model_classification_with_keypoint(
    KEYPOINT_MODELS, CLASSIFICATION_MODELS, ckpt_path
):
    kpt_model = importlib.import_module(f"models.{KEYPOINT_MODELS}")
    kpt_model = kpt_model.MyLightningModule(
        params, num_classes=18
    )  # locals()[f'{MODEL_NAME}']()
    kpt_model = kpt_model.load_from_checkpoint(ckpt_path, params=params)
    kpt_model.eval()
    model = importlib.import_module(f"models.{CLASSIFICATION_MODELS}")
    model = model.MyLightningModule(kpt_model)
    return model


def train_for_model_finetune_classification(
    default_root_dir, train_dataloader, test_dataloader, model
):
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
    return trainer


# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# from models.modelA import MyLightningModule as modelA

# from models.modelB import MyLightningModule as modelB
# from models.modelD import MyLightningModule as modelD
# from models.modelH import MyLightningModule as modelH


# from models.modelE import MyLightningModule as modelE
def train_and_evaluate_finetune_parameterized_by_config(
    MODEL_NAME,
    PRETRAIN_MODEL,
    model,
    default_root_dir,
    train_dataloader,
    test_dataloader,
    PARAM_NAME,
):
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
        max_epochs=300,
        min_epochs=300,
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
        # f.write(str(params))
        # f.write("\n\n\n")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def create_dataloader_finetune_parameterized_by_config(MODEL_NAME, BATCH_SIZE):
    test_dataset = SQLJointsDataset(train=False)

    WITH_FILTERED = True
    if MODEL_NAME == "model0":
        # test_dataset = SQLJointsDataset(train=False)
        SQLJointsDataset.TABLE_NAME = "openpose_annotation_old_data"
        train_dataset = SQLJointsDataset(train=True)
    elif WITH_FILTERED == True:
        SQLJointsDataset.TABLE_NAME = "openpose_annotation_03_18_with_quilt_filtered"
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

    return default_root_dir, train_dataloader, test_dataloader
