import csv
import os
import pytorch_lightning as pl
import importlib
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
