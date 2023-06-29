import pytorch_lightning as pl
import torch
from torch import nn

from .hyperparameters import *


# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
class ClassificationModule(pl.LightningModule):
    """
    A base class for classification models, which sets up the training, validation steps, optimizers, loss functions, and hyperparameters.
    """

    def __init__(self):
        super().__init__()
        self.net = None
        # self.conv = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,padding=1).cuda()
        self.classification_loss = nn.CrossEntropyLoss()  # label_smoothing=0.001)
        # self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = input.float()
        input = self.conv(input)
        classify = self.net(input)
        return classify

    def training_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
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

    def validation_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
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
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
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
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        self.lr = lr
        self.l2 = l2
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2
        )

        return [self.optimizer]

    def loss_calculation(self, batch):
        _, _, _, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        # target = target[:,0]#,:]

        # regression_loss = self.joint_loss(regress,target,target_weight) * 1000
        class_target = meta["posture"]
        classification_loss = self.classification_loss(
            classify, class_target
        )  # , y.argmax(dim=1)
        loss = classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        # _, joint_acc, cnt, pred = accuracy(regress.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        return loss, None, class_acc

    def get_batch_output(self, batch):
        """get_batch_output _summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        input, target, target_weight, meta = batch
        # print(meta.keys())
        # joints = meta["joints"].flatten(start_dim=1)
        regress, classify = self(input)  # , joints)
        return {"classify": classify, "regress": regress}
