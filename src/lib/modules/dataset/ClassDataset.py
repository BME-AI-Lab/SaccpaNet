# %% [markdown]
# %load_ext sql

# %% [markdown]
# %sql mysql+pymysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu

# %%
import numpy as np
import sqlalchemy
from .common import SQL_indexer

#image_index = SQL_indexer("SELECT a.depth_array FROM data_06_15_images as a WHERE a.index={}","mysql+pymysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu")

# %%
#import pymysql

# %%

#np.loads(image_index[1])

# %%
#len(image_index[1].first().keys())

# %%
import torch
from torch.utils.data import Dataset
class PandasDataset(Dataset):
        def __init__(self, df,x,y=None,index = "index",transform =None,target_transform = None , co_transform = None, *args,**kwargs):
                super().__init__()#*args,**kwargs)
                self.df = df
                self.x = x
                self.y = y
                self.transform= transform
                self.target_transform = target_transform
                self.co_transform = co_transform
                self.index = index
        def __len__(self):
                return len(self.df)
        def __getitem__(self,key):
                if callable(self.x):
                        x = self.x(self.df.iloc[key])
                else:
                        x = self.df.iloc[key][self.x]
                if callable(self.y):
                    y = self.y(self.df.iloc[key])
                else:
                    y = self.df.iloc[key][self.y]
                if self.transform:
                        x = self.transform(x)
                if self.target_transform: 
                        y = self.target_transform(y)
                if self.co_transform:
                        x,y = self.co_transform(x,y)
                return x,y
                    
            

# %%


# %%
#from functools import lru_cache
from common import ResolveImage,fill_hole
#resolve_img = ResolveImage()

# %%
import pandas as pd
df = pd.read_sql("SELECT * FROM data_06_15_annotations","mysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu")

# %%
row = df.iloc[0]
#resolve(row,0).shapea

# %%
groups = dict()
groups[2] = {'r': 0, 'o': 1, 'y': 1, 'g': 1, 'c': 1, 'b': 0, 'p': 0}
groups[4] = {'r': 0, 'o': 1, 'y': 1, 'g': 2, 'c': 2, 'b': 3, 'p': 3}
groups[6] = {'r': 0, 'o': 1, 'y': 2, 'g': 3, 'c': 4, 'b': 5, 'p': 5}
groups[7] = {'r': 0, 'o': 1, 'y': 2, 'g': 3, 'c': 4, 'b': 5, 'p': 6}
m = groups[7]
df["posture_group"] = df["posture"].map(lambda x: m[x])

# %%
train_df = df[df["subset"]=="train"]
test_df = df[df["subset"]=="test"]
#dataset combinative expanding
expanded_train_df = train_df.merge(train_df,on=["posture","subset","subject_number"])#"annotation_file","x1","x2","y1","y2","bag_file",
expanded_train_df = expanded_train_df.rename(columns={"x1_x":"x1","x2_x":"x2","y1_x":"y1","y2_x":"y2","annotation_file_x":"annotation_file"})
expanded_train_df = expanded_train_df[expanded_train_df["effect_x"]<=expanded_train_df["effect_y"]] #left triangle filter

print(expanded_train_df.head(50))
# %%
input_size=(192,256) #updated for posture estimation, original 224*224

# %%
import cv2
from random import random
class PairedData:
    def __init__(self,shape = input_size,probability=0.5):
        self.shape = shape
        self.probability = probability
        self.resolve_img = ResolveImage()
    
    def __call__(self,*args,**kwargs):
        return self._resolve_paired_data(*args,**kwargs)
    
    def _resolve_paired_data(self,row,shape=None,probability = None):
        u = np.random.rand(1) #u ~Uniform(0,1)
        img1_index = row.index_x
        img2_index = row.index_y
        img1 = self.resolve_img(row,img1_index,self.shape)
        img2 = self.resolve_img(row,img2_index,self.shape)
        #assert y1 == y2
        
        #img1 = cv2.resize(img1,shape)
        #img2 = cv2.resize(img2,shape)
        if random()<self.probability:
            self.probability +=1-0.02
            
            return u* img1+ (1-u)*img2
        else:
            if random()<=0.5:
                return img1
            else:
                return img2
#if __name__ == "__main__":
                   
resolve_paired_data = PairedData()
    

# %%
#import matplotlib.pyplot as plt
#if __name__ == "__main__":
merged = resolve_paired_data(expanded_train_df.iloc[1],input_size)

# %%
#expanded_train_df.iloc[1]

# %%
"""
import seaborn as sns
mean = []
std = []
for i in range(100):
    merged = resolve_paired_data(expanded_train_df.iloc[i])
    mean.append(np.mean(merged)),std.append(np.std(merged))
plt.imshow(merged)
"""
# %%
#np.mean(merged),np.std(merged)

# %%
#np.median(mean),np.median(std)

# %%
import torchvision
from sklearn.utils import class_weight
# y_train = expanded_train_df["posture"]
# n_classes = len(np.unique(y_train))
# class_weights = class_weight.compute_class_weight('balanced',
#                     np.unique(y_train),
#                     y_train)
target_transform = torchvision.transforms.Compose([
    lambda x:torch.LongTensor([x]),]) # or just torch.tensor
    #lambda x:F.one_hot(x,n_classes)])

# %%
import torchvision
from torchvision.transforms import *
from PIL import Image
#
train_transforms = Compose([
    
    lambda x: x.astype(np.float),#[:,:],
    #lambda x:print(x.shape),
    #torch.Tensor,
    ToTensor(),
    lambda x: x.cuda(1),
    #torch.nn.Sequential(
    torch.nn.Sequential(
    Normalize(1590.6,60.29),
    #RandomRotation(5),
    #RandomCrop(0.05,padding_mode="edge"),
    RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.95,1.05)),
    ).cuda(1)
    #,
    
])
"""
    torch.nn.Sequential(
    Normalize(1590.6,60.29),
    #RandomRotation(5),
    #RandomCrop(0.05,padding_mode="edge"),
    RandomAffine(degrees=10,translate=(0.1,0.1),scale=(0.95,1.05)),
    )
    """
test_transforms = torchvision.transforms.Compose([
     lambda x: x.astype(np.float),
    #lambda x: x[:,:],
    #lambda x:print(x.shape),
    #torch.Tensor,
    ToTensor(),
    Normalize((1590.6,),(60.29,)),
    #lambda x:print(x.shape),
])


# %%
torchvision.__version__

# %% [markdown]
# 

# %% [markdown]
# len(test_df)

# %%

if __name__=="__main__":
    train_dataset = PandasDataset(expanded_train_df,resolve_paired_data,y="posture_group_x",
                                transform=train_transforms,target_transform=target_transform)
    resolve_img = ResolveImage()
    train_dataset = PandasDataset(train_df,resolve_img,y="posture_group",
                                transform=train_transforms,target_transform=target_transform)
    test_dataset = PandasDataset(test_df,resolve_img,y="posture_group",
                                transform=test_transforms,target_transform=target_transform)

    # %%
    #len(train_dataset),len(test_dataset)

    # %%
    batch_size = 8

    from torch.utils.data import DataLoader
    #if __name__ == "__main__":
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=0)#,pin_memory=True)# ,prefetch_factor=128)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,  num_workers=0)#,pin_memory=True)