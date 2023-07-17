from torch.utils.data.dataloader import DataLoader

from configs.dataset_config import DATALOADER_WORKERS
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset


def create_test_dataloader(BATCH_SIZE, WITH_QUILT=True):
    test_dataset = SQLJointsDataset(is_train=False, test=True, all_quilt=WITH_QUILT)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
        persistent_workers=DATALOADER_WORKERS,
    )

    return test_dataloader


def create_train_dataloader(BATCH_SIZE, WITH_QUILT=True):
    train_dataset = SQLJointsDataset(is_train=True, all_quilt=WITH_QUILT)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
        persistent_workers=DATALOADER_WORKERS,
    )

    return train_dataloader


def create_dataloaders(BATCH_SIZE):
    train_dataloader = create_train_dataloader(BATCH_SIZE)
    val_dataloader = create_validation_dataloader(BATCH_SIZE, WITH_QUILT=True)
    return train_dataloader, val_dataloader


def create_validation_dataloader(BATCH_SIZE, WITH_QUILT):
    test = False
    if WITH_QUILT:
        val_dataset = SQLJointsDataset(
            is_train=False, mixed=False, all_quilt=True, test=test
        )
    else:
        val_dataset = SQLJointsDataset(
            is_train=False, mixed=False, all_quilt=False, test=test
        )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return val_dataloader


if __name__ == "__main__":
    print(len(create_train_dataloader(16)))
