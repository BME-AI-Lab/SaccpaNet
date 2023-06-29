from torch import nn

from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.networks.SaccpaNet import SaccpaNet
from models.RegressionBase import RegressionModule


class MyLightningModule(RegressionModule):
    def __init__(self, params, num_joints=18):
        super().__init__()
        self.hparams.update(params)
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        )
        self.net = SaccpaNet(params, num_joints=num_joints)
        self.preNorm = nn.BatchNorm2d(num_features=1)

    def forward(self, input):
        input = input.float()
        input = self.preNorm(input)
        input = self.init_conv(input)
        regress = self.net(input)
        return regress, None
