


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
from lib.modules.dataset.SQLJointsDataset import SQLJointsDataset
from lib.modules.core.function import accuracy
from lib.modules.core.loss import JointsMSELoss
from lib.modules.modules import Network
from .config import *

# %%
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b0')
class RegressionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,padding=1).cuda()
        self.net = Network(1,18)
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()#label_smoothing=0.001)
        #self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        
    def forward(self, input):
        input = input.float().cuda()
        regress,classify = self.net(input)
        return regress,classify

    def get_batch_output(self,batch):
        input, target, target_weight, meta = batch
        #print(meta.keys())
        joints = meta["joints"].flatten(start_dim=1)
        regress, classify = self(input,joints)
        return {"classify":classify,"regress":regress}

    def training_step(self, batch, batch_idx):
        loss, acc,class_acc = self.loss_calculation(batch)
        #acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('train_joint_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss",loss)
        
        return {"loss":loss,"train_loss":loss,"train_joint_acc":acc}#

    def loss_calculation(self, batch):
        input, target, target_weight, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        regress = result["regress"]
        
        regression_loss = self.joint_loss(regress,target,target_weight) * 1000 
        class_target = meta['posture']
        classification_loss = self.classification_loss(classify,class_target)#, y.argmax(dim=1)
        loss = regression_loss + classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        _, joint_acc, cnt, pred = accuracy(regress.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        return loss,joint_acc,class_acc
    
    def validation_step(self, batch, batch_idx):
        loss, acc,class_acc = self.loss_calculation(batch)
        self.log('val_joint_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss",loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss":loss,"val_joint_acc":acc}
    
    def test_step(self, batch, batch_idx):
        loss, acc ,class_acc = self.loss_calculation(batch)
        self.log('test_joint_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss":loss,"test_joint_acc":acc}
    
    def configure_optimizers(self):
        lr = lr#0.02#0.00002c
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=l2)#,momentum=0.9#5e-5
        """
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,weight_decay=5e-5,momentum=0.9)#,momentum=0.9
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                                        self.optimizer, max_lr=lr,base_lr =1e-5,step_size_up =50,) #base_lr 1e-5
                                        #anneal_strategy='linear', div_factor=10000,
                                        #steps_per_epoch=int((len(train_dataset)/batch_size)),
                                        #epoch
        sched = {
            'scheduler': self.scheduler,
            'interval': 'step',
        }
        #"""
        return [self.optimizer]#, [sched]