


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

# %%
import pytorch_lightning as pl
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from .classification import ClassificationModule

#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b0')
class MyLightningModule(ClassificationModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,padding=1).cuda()
        self.net = models.efficientnet_b4(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.dense = nn.Linear(1054,256,bias=True)#input_size[0]*input_size[1]
        self.end = nn.Sequential(
            nn.ELU(),
            nn.Linear(256,256,bias=True),        
        )
        self.dense3 = nn.Linear(256,7)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        self.alt_dense = nn.Linear(1000,7)
        
    def forward(self, x,d):
        #print(x.shape)
        #print(x)
        #x,d = x
        #print(x)
        x = x.float()#.cuda()
        d = d.float()
        x = self.conv(x)
        x = self.net(x)
        #print(x.shape)
        x = x.flatten(start_dim=1)
        #x = self.dropout(x)
        #alternative_out = self.alt_dense(x)
        x = torch.cat((x,d),dim=1)
        #d = self.tanh(d)
        x = self.dense(x)
        #print(d)
        x = self.end(x)
        #x = self.dense2(self.relu(x))
        x = self.dense3(x)
        #x = self.softmax(x)
        
        return x#,alternative_out
    def get_batch_output(self,batch):
        input, target, target_weight, meta = batch
        #print(meta.keys())
        joints = meta["joints"].flatten(start_dim=1)
        classify = self(input,joints)
        return {"classify":classify}

        
    def loss_calculation(self, batch):
        _, _, _, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        #target = target[:,0]#,:]
        
        #regression_loss = self.joint_loss(regress,target,target_weight) * 1000 
        class_target = meta['posture']
        classification_loss = self.classification_loss(classify,class_target)#, y.argmax(dim=1)
        loss =  classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        # _, joint_acc, cnt, pred = accuracy(regress.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        return loss,None,class_acc