import torch
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from .hyperparameters import lr, l2
import pytorch_lightning as pl
from torch import nn


class RegressionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=1
        ).cuda()
        self.net = None
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, input):
        input = input.float().cuda()
        regress, classify = self.net(input)
        return regress, classify

    def get_batch_output(self, batch):
        input, target, target_weight, meta = batch
        # print(meta.keys())
        # joints = meta["joints"].flatten(start_dim=1)
        regress, classify = self(input)  # , joints)
        return {"classify": classify, "regress": regress}

    def training_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
        self.log(
            "train_joint_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc",
            class_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train_loss", loss)

        return {"loss": loss, "train_loss": loss, "train_joint_acc": acc}  #

    def loss_calculation(self, batch):
        input, target, target_weight, meta = batch
        result = self.get_batch_output(batch)
        regress = result["regress"]
        regression_loss = self.joint_loss(regress, target, target_weight) * 1000
        loss = regression_loss
        _, joint_acc, cnt, pred = accuracy(
            regress.detach().cpu().numpy(), target.detach().cpu().numpy()
        )
        return loss, joint_acc, 0

    def validation_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
        self.log(
            "val_joint_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            class_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "val_joint_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
        self.log(
            "test_joint_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            class_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_joint_acc": acc}

    def configure_optimizers(self):
        self.lr = lr
        self.l2 = l2
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2
        )
        return [self.optimizer]  # , [sched]
