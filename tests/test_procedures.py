from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset
from lib.procedures.dataloaders_procedure import *


def _iterate_through_dl(dl):
    for i in dl:
        assert i


def test_train_dataloader():
    dl = create_train_dataloader(1)
    _iterate_through_dl(dl)
    dl = create_train_dataloader(8)
    _iterate_through_dl(dl)


def test_test_dataloader():
    dl = create_test_dataloader(1)
    _iterate_through_dl(dl)
    dl = create_test_dataloader(8)
    _iterate_through_dl(dl)


def test_validation_dataloader():
    dl = create_validation_dataloader(1, False, False)
    _iterate_through_dl(dl)
    dl = create_validation_dataloader(1, True, False)
    _iterate_through_dl(dl)
    dl = create_validation_dataloader(1, True, True)
    _iterate_through_dl(dl)
    dl = create_validation_dataloader(1, False, True)
    _iterate_through_dl(dl)
