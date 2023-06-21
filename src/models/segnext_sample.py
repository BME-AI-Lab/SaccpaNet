# %%
import importlib

import torch
import torchvision.models

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from torch import nn

from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss

# %%
from lib.networks.SegNext import SegNextU

from .base_module import ClassificationModule


# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
class MyLightningModule(ClassificationModule):
    def __init__(self, params, num_classes=18):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=19, out_channels=3, kernel_size=3, padding=1
        ).cuda()
        self.hparams.update(params)
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        ).cuda()
        # self.fix_padding_conv = nn.Conv2d(
        #    in_channels=18, out_channels=18, kernel_size=3, padding="valid"
        # ).cuda()
        self.net = SegNextU(params, num_classes=num_classes)  # (num_classes = 18
        # self.classify_net = eca_resnet50(num_classes=7)#(num_classes = 7)
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()  # label_smoothing=0.001)
        # self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()

        self.preNorm = nn.BatchNorm2d(num_features=1)
        # self.deconv1 = nn.ConvTranspose2d()

    def forward(self, input):
        input = input.float().cuda()
        input = self.preNorm(input)
        x = self.init_conv(input)
        # print(x.shape)
        # x=input
        regress = self.net(x)  # [:,-1]#["out"]
        # regress = self.fix_padding_conv(regress)
        # print(regress.shape,input.shape)
        # x = torch.concat([input,regress],dim=1)
        # x = self.conv(x)
        # classify = self.classify_net(x)
        return regress, None  # classify

    def training_step(self, batch, batch_idx):
        loss, acc, class_acc = self.loss_calculation(batch)
        # acc = (y_hat.argmax(dim=-1) == y).float().mean()
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
        stacked_input = input  # torch.cat((input,target),dim=1)
        regress, classify = self(stacked_input)
        return {"classify": classify, "regress": regress}

    def loss_calculation(self, batch):
        input, target, target_weight, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        regress = result["regress"]
        # target = target[:,0]#,:]

        regression_loss = self.joint_loss(regress, target, target_weight) * 1000
        # class_target = meta["posture"]
        # classification_loss = self.classification_loss(classify,class_target)#, y.argmax(dim=1)
        loss = regression_loss  # + 0*classification_loss
        # class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        _, joint_acc, cnt, pred = accuracy(
            regress.detach().cpu().numpy(), target.detach().cpu().numpy()
        )
        return loss, joint_acc, 0  # class_acc

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
