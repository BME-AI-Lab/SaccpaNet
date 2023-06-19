



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
from lib.modules.eca import eca_resnet50
from glob import glob

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
        self.net = eca_resnet50(num_classes=7,initial_layer=3)#Network(1,18)
        #self.joint_loss = JointsMSELoss(use_target_weight=True)
        self.classification_loss = nn.CrossEntropyLoss()#label_smoothing=0.001)
        #self.dense = nn.Linear(1000,7)#input_size[0]*input_size[1]
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()
        self.lr=0.0001
        
    def forward(self, input):
        input = input.float().cuda()
        input = self.conv(input)
        classify = self.net(input)
        return classify

    def get_batch_output(self,batch):
        input, target, target_weight, meta = batch
        stacked_input = input#torch.cat((input,target),dim=1)
        classify = self(stacked_input)
        return {"classify":classify}

        
    def loss_calculation(self, batch):
        _, _, _, meta = batch
        result = self.get_batch_output(batch)
        classify = result["classify"]
        
        #regression_loss = self.joint_loss(regress,target,target_weight) * 1000 
        class_target = meta['posture']
        classification_loss = self.classification_loss(classify,class_target)#, y.argmax(dim=1)
        loss =  classification_loss
        class_acc = (classify.argmax(dim=-1) == class_target).float().mean()
        # _, joint_acc, cnt, pred = accuracy(regress.detach().cpu().numpy(),
        #                                  target.detach().cpu().numpy())
        return loss,None,class_acc
    
if __name__ == "__main__":
    BATCH_SIZE = 32
    train_dataset = SQLJointsDataset(train=True)
    test_dataset = SQLJointsDataset(train=False)
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
    VERSION = 11
    CHECKPOINT = f"./lightning_logs/version_{VERSION}/checkpoints/"
    ckpt_file = glob(CHECKPOINT+"*.ckpt")[0]
    model = MyLightningModule()
    #model = model.load_from_checkpoint(ckpt_file)
    #model = MyLightningModule()
    trainer = pl.Trainer(gpus=[0,1],
        amp_level="O2", accelerator='dp',amp_backend='apex',
        max_epochs=-1,min_epochs=500,
        )#gpus=1, accelerator='dp',

    trainer.tune(model,train_dataloader)
    #model = model.load_from_checkpoint(ckpt_file)
    trainer.fit(model, train_dataloader, test_dataloader)

    # %%
    #(model.net._avg_pooling.output_size)
    # %%
    x = trainer.test(model,test_dataloader,verbose=True)

    # %%
    x
