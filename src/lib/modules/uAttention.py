from torch import nn
from collections import OrderedDict

def HeightWidthConv(self,channels,kernel_size,stride,padding,dilation,padding_mode="circular"):
        #assert in_channels == out_channels
        #assert kernel_size == 3
        convHeight = nn.Conv2d(channels,channels,(1,kernel_size),stride,padding,dilation,groups=channels,padding_mode=padding_mode)
        convWidth = nn.Conv2d(channels,channels,(1,kernel_size),stride,padding,dilation,groups=channels,padding_mode=padding_mode)
        return nn.Sequential(OrderedDict(
            ('convHeight',convHeight),
            ('convWidth',convWidth)
        ))

class BlockConstruction(nn.Module):
    def __init__(self,networks):
        blocks = []
        for i in :
            network = 