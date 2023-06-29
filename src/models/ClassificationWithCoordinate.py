import torch
import torchvision.models as models
from torch import nn

from lib.modules.core.loss import JointsMSELoss
from lib.modules.core.SpatialSoftArgmax2d import SpatialSoftArgmax2d

from .ClassificationBase import ClassificationModule
from .hyperparameters import l2, lr


class MyLightningModule(ClassificationModule):
    def __init__(self, JointNetwork, ClassifyNetwork):
        num_classes = 7
        super().__init__()
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        )
        self.coordinate_net = JointNetwork
        # Disable training to coordinate network
        for param in self.coordinate_net.parameters():
            param.requires_grad = False
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()
        self.classify_net = ClassifyNetwork
        self.loss = nn.CrossEntropyLoss()
        self.dense = nn.Linear(1036, 256, bias=True)
        self.end = nn.GELU()
        self.dense3 = nn.Linear(256, num_classes)
        self.preNorm = nn.BatchNorm2d(num_features=1)
        self.spatialSoftArgMax = SpatialSoftArgmax2d(normalized_coordinates=True)

    def forward(self, input):
        input = input.float()
        input = self.preNorm(input)
        x = self.init_conv(input)
        with torch.no_grad():
            regress = self.coordinate_net(input)
            coordinates = self.spatialSoftArgMax(regress[0])  # TBD: double check
            coordinates = coordinates.flatten(start_dim=1)

        x = self.classify_net(x)
        x = x.flatten(start_dim=1)
        x = torch.cat((x, coordinates), dim=1)
        x = self.dense(x)
        x = self.end(x)
        classify = self.dense3(x)
        return None, classify
