import torch
import torchvision.models as models
from torch import nn

from lib.modules.core.loss import JointsMSELoss
from lib.modules.core.SpatialArgmax2d import HardArgmax2d

from .ClassificationBase import ClassificationModule
from .hyperparameters import l2, lr

torch.autograd.set_detect_anomaly(True)


class MyLightningModule(ClassificationModule):
    def __init__(self, JointNetwork, ClassifyNetwork):
        num_classes = 7
        super().__init__()
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        ).float()
        self.coordinate_net = JointNetwork
        # Disable training to coordinate network
        for param in self.coordinate_net.parameters():
            param.requires_grad = False
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()
        self.classify_net = ClassifyNetwork  # .double()
        self.loss = nn.CrossEntropyLoss()  # .float()
        self.dense = nn.Linear(1036, 256, bias=True)  # .double()
        self.end = nn.GELU()  # .float()
        self.dense3 = nn.Linear(256, num_classes)  # .double()
        self.preNorm = nn.BatchNorm2d(num_features=1)  ##.float()
        self.spatialArgMax = HardArgmax2d(normalized_coordinates=True)

    def forward(self, input):
        input = input.float()
        input = self.preNorm(input)
        x = self.init_conv(input)
        # x = x.double()
        x = self.classify_net(x)
        # x = x.float()
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            regress = self.coordinate_net(input)
            coordinates = self.spatialArgMax(regress[0])  # TBD: double check
            coordinates = coordinates.flatten(start_dim=1)
            # coordinates = coordinates.double()
        # x = x.double()
        x = torch.cat((x, coordinates), dim=1)
        x = self.dense(x)
        x = self.end(x)
        classify = self.dense3(x)
        return None, classify
