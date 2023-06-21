from models.regression import RegressionModule
from torch import nn
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.networks.SegNext import SegNextU


class MyLightningModule(RegressionModule):
    def __init__(self, params, num_classes=18):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=19, out_channels=3, kernel_size=3, padding=1
        ).cuda()
        self.hparams.update(params)
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        ).cuda()
        self.net = SegNextU(params, num_classes=num_classes)
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        self.preNorm = nn.BatchNorm2d(num_features=1)

    def forward(self, input):
        input = input.float().cuda()
        input = self.preNorm(input)
        x = self.init_conv(input)
        regress = self.net(x)
        return regress, None

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

    def get_batch_output(self, batch):
        input, target, target_weight, meta = batch
        regress, classify = self(input)
        return {"classify": classify, "regress": regress}

    def loss_calculation(self, batch):
        # T TBD abstract out the loss calculation for regression models
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
