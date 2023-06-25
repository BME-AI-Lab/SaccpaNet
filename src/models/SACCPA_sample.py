from models.RegressionBase import RegressionModule
from torch import nn
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.networks.SegNext import SaccpaNet


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
        self.net = SaccpaNet(params, num_classes=num_classes)
        self.preNorm = nn.BatchNorm2d(num_features=1)

    def forward(self, input):
        input = input.float().cuda()
        input = self.preNorm(input)
        x = self.init_conv(input)
        regress = self.net(x)
        return regress, None
