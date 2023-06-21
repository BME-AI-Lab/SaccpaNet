# %%
import torch
import torchvision.models
import importlib

importlib.reload(torchvision)
torch.__version__

# %%
#!pip uninstall efficientnet_pytorch -y

# %%
#!git clone https://github.com/lukemelas/EfficientNet-PyTorch

# %%
from lib.modules.core.loss import JointsMSELoss


# %%
import torchvision.models as models
from torch import nn
from .base_module import ClassificationModule

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
from lib.modules.core.SpatialSoftArgmax2d import SpatialSoftArgmax2d


class MyLightningModule(ClassificationModule):
    def __init__(self, JointNetwork):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=19, out_channels=3, kernel_size=3, padding=1
        ).cuda()
        self.init_conv = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=1, padding=0
        ).cuda()
        self.fix_padding_conv = nn.Conv2d(
            in_channels=18, out_channels=18, kernel_size=3, padding="valid"
        ).cuda()
        self.net = JointNetwork  # S()
        for param in self.net.parameters():
            param.requires_grad = False
        # self.classify_net = eca_resnet50(num_classes=7)#(num_classes = 7)
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()  # label_smoothing=0.001)
        # self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        self.classify_net = models.efficientnet_b4(pretrained=True)  # eca_resnet50
        self.loss = nn.CrossEntropyLoss()
        self.dense = nn.Linear(1036, 256, bias=True)  # input_size[0]*input_size[1]
        self.end = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, 256, bias=True),
        )
        self.dense3 = nn.Linear(256, 7)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        self.alt_dense = nn.Linear(1000, 7)

        self.preNorm = nn.BatchNorm2d(num_features=1)
        self.softArgMax = SpatialSoftArgmax2d(normalized_coordinates=True)

        # self.deconv1 = nn.ConvTranspose2d()

    def coordinateFromHM(self, hm):
        return self.softArgMax(hm)

    def forward(self, input):
        input = input.float().cuda()
        input = self.preNorm(input)
        x = self.init_conv(input)
        # print(x.shape)
        # x=input
        with torch.no_grad():
            regress = self.net(input)  # [:,-1]#["out"]
            coordinates = self.coordinateFromHM(regress[0])
            coordinates = coordinates.flatten(start_dim=1)
        # regress = self.fix_padding_conv(regress)
        # print(regress.shape,input.shape)
        # x = torch.concat([input,regress],dim=1)
        x = self.classify_net(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # x = self.dropout(x)
        # alternative_out = self.alt_dense(x)
        x = torch.cat((x, coordinates), dim=1)
        # d = self.tanh(d)
        x = self.dense(x)
        # print(d)
        x = self.end(x)
        # x = self.dense2(self.relu(x))
        classify = self.dense3(x)
        return None, classify  # classify

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

        # regression_loss = self.joint_loss(regress,target,target_weight) * 1000
        class_target = meta["posture"]
        classification_loss = self.classification_loss(
            classify, class_target
        )  # , y.argmax(dim=1)
        loss = classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        # _, joint_acc, cnt, pred = accuracy(regress.detach().cpu().numpy(),
        #                                 target.detach().cpu().numpy())
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
        self.lr = 0.001
        self.l2 = 5e-5
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2
        )  # ,momentum=0.9#5e-5
        """
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,weight_decay=5e-5,momentum=0.9)#,momentum=0.9
        
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                                        self.optimizer, max_lr=self.lr,base_lr =1e-8,step_size_up =50,) #base_lr 1e-5
                                        #anneal_strategy='linear', div_factor=10000,
                                        #steps_per_epoch=int((len(train_dataset)/batch_size)),
                                        #epoch
        sched = {
            'scheduler': self.scheduler,
            'interval': 'step',
        }
        #"""
        return [self.optimizer]  # , [sched]
