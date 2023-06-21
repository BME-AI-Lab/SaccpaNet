# %%
import torch
import torchvision.models
import importlib
from .hyperparameters import lr, l2

importlib.reload(torchvision)
torch.__version__
from lib.modules.core.loss import JointsMSELoss
import torchvision.models as models
from torch import nn
from .base_module import ClassificationModule
from lib.modules.core.SpatialSoftArgmax2d import SpatialSoftArgmax2d


class MyLightningModule(ClassificationModule):
    def __init__(self, JointNetwork, ClassifyNetwork):
        super().__init__()
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        ).cuda()
        self.coordinate_net = JointNetwork
        # Disable training to coordinate network
        for param in self.coordinate_net.parameters():
            param.requires_grad = False
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()
        self.classify_net = ClassifyNetwork
        self.loss = nn.CrossEntropyLoss()
        self.dense = nn.Linear(1036, 256, bias=True)
        self.end = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 256, bias=True),
        )
        self.dense3 = nn.Linear(256, 7)
        self.preNorm = nn.BatchNorm2d(num_features=1)
        self.spatialSoftArgMax = SpatialSoftArgmax2d(normalized_coordinates=True)

    def coordinateFromHM(self, hm):
        """Generate Coordinate from Heatmap

        Args:
            hm (N*C*H*W): C numebr of coordinate heatmpa layer

        Returns:
            (N*C*2): Number of coordinates in x,y
        """
        return self.spatialSoftArgMax(hm)

    def forward(self, input):
        input = input.float().cuda()
        input = self.preNorm(input)
        x = self.init_conv(input)
        with torch.no_grad():
            regress = self.coordinate_net(input)
            coordinates = self.coordinateFromHM(regress[0])
            coordinates = coordinates.flatten(start_dim=1)

        x = self.classify_net(x)
        x = x.flatten(start_dim=1)
        x = torch.cat((x, coordinates), dim=1)
        x = self.dense(x)
        x = self.end(x)
        classify = self.dense3(x)
        return None, classify

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
        stacked_input = input  # torch.cat((input,target),dim=1)
        regress, classify = self(stacked_input)
        return {"classify": classify, "regress": regress}

    def loss_calculation(self, batch):
        input, target, target_weight, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        regress = result["regress"]
        class_target = meta["posture"]
        classification_loss = self.classification_loss(classify, class_target)
        loss = classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()

        return loss, 0, class_acc

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
        return [self.optimizer]
