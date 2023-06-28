from torch.utils.data.dataloader import DataLoader

from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset


def create_test_dataloader(BATCH_SIZE):
    test_dataset = SQLJointsDataset(train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=4,
    )

    return test_dataloader


def create_train_dataloader(BATCH_SIZE):
    train_dataset = SQLJointsDataset(train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=4,
    )

    return train_dataloader


def create_dataloaders(BATCH_SIZE):
    train_dataloader = create_train_dataloader(BATCH_SIZE)
    test_dataloader = create_test_dataloader(BATCH_SIZE)
    return train_dataloader, test_dataloader


def create_validation_dataloader(BATCH_SIZE, WITH_QUILT, VALIDATION):
    if WITH_QUILT:
        val_dataset = SQLJointsDataset(
            train=False, mixed=False, all_quilt=True, validation=VALIDATION
        )
        quilt_conditions = "all_quilts"
    else:
        val_dataset = SQLJointsDataset(
            train=False, mixed=False, all_quilt=False, validation=VALIDATION
        )
        quilt_conditions = "no_quilts"
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return val_dataloader
