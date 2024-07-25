from torch import nn

from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.networks.SaccpaNetAblationNetStructureParamsAtrousAttention import SaccpaNet
from models.RegressionBase import RegressionModule


class MyLightningModule(RegressionModule):
    def __init__(self, params, num_joints=18, uattention_arms=3, uattention_atrous=3):
        super().__init__()
        self.hparams.update(params)
        self.net = SaccpaNet(
            params,
            num_joints=num_joints,
            uattention_arms=uattention_arms,
            uattention_atrous=uattention_atrous,
        )
        self.preNorm = nn.BatchNorm2d(num_features=1)

    def forward(self, input):
        input = input.float()
        input = self.preNorm(input)
        regress = self.net(input)
        return regress, None
