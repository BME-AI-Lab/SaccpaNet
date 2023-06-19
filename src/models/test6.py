


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
from lib.modules.test6_attentionUnet import BranchedAttU_Net
from lib.modules.eca import eca_resnet50


# %%
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from .classification import  ClassificationModule
#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b0')
class MyLightningModule(ClassificationModule):
    def __init__(self):
        super().__init__()
        self.net = BranchedAttU_Net(1,18)
        #self.classify_net = eca_resnet50(num_classes=7)#(num_classes = 7)
        self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()#label_smoothing=0.001)
        #self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        #self.dropout = nn.Dropout()
        #self.softmax = nn.Softmax()
        
    def forward(self, input):
        input = input.float().cuda()
        regress,classify = self.net(input)
        return regress,classify

    def training_step(self, batch, batch_idx):
        loss, acc,class_acc = self.loss_calculation(batch)
        #acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('train_joint_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', class_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss",loss)
        
        return {"loss":loss,"train_loss":loss,"train_joint_acc":acc}#
    
    def get_batch_output(self,batch):
        input, target, target_weight, meta = batch
        stacked_input = input#torch.cat((input,target),dim=1)
        regress,classify = self(stacked_input)
        return {"classify":classify,"regress":regress}

    def loss_calculation(self, batch):
        input, target, target_weight, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        regress = result["regress"]
        #target = target[:,0]#,:]
        
        regression_loss = self.joint_loss(regress,target,target_weight) * 1000 
        class_target = meta['posture']
        print(classify.shape,class_target.shape)
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
    
