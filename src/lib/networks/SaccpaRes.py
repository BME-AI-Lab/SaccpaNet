import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class UAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #remark: initial search without 1x1 
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding="same", dilation=1, groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", dilation=1, groups=dim)
        #self.conv0_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 3), padding="same", dilation=2, groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", dilation=2, groups=dim)
        #self.conv1_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)
        
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 3), padding="same", dilation=5, groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", dilation=5, groups=dim)
        #self.conv2_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)
        
        self.conv3_1 = nn.Conv2d(dim, dim, (1, 3), padding="same", dilation=7, groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", dilation=7,groups=dim)
        #self.conv3_3 = nn.Conv2d(dim, dim, (1, 1), padding="same", dilation=1, groups=dim)
        
        #self.conv1x1 = nn.Conv2d(dim, dim, (1, 1), padding="same", groups=dim)
        #self.down = nn.Conv2d(dim, dim, (1, 3), padding="same", groups=dim)
        #self.conv3_2 = nn.Conv2d(dim, dim, (3, 1), padding="same", groups=dim)
    
    def forward(self,x):
        attn = x.clone()
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        #attn_0 = self.conv0_3(attn_0)
        attn = attn+attn_0

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        #attn_1 = self.conv1_3(attn_1)
        attn = attn + attn_1

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        #attn_2 = self.conv2_3(attn_2)
        attn = attn + attn_2
        
        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        #attn_3 = self.conv3_3(attn_3)
        attn = attn + attn_3
        
        #attn = self.conv1x1(attn)
        
        #attn = attn + attn_0 + attn_1 + attn_2
        return attn 

class UAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        image_size = 512
        reduction = 4,8,16,32
        
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.down1_0 = UAttentionLayer(dim)
        self.down1_1 = nn.AvgPool2d(2)
        self.down2_0 = UAttentionLayer(dim)
        self.down2_1 = nn.AvgPool2d(2)
        self.down3_0 = UAttentionLayer(dim)
        
        

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.down1_0(attn)
        x = self.down1_1(attn_0.clone())

        attn_1 = self.down2_0(x)
        x = self.down2_1(attn_1.clone())

        attn_2 = self.down3_0(x)
        #attn_2 = self.down3_1(attn_2)
        attn_1 = nn.functional.interpolate(attn_1.clone(),u.shape[2:],mode="bilinear")
        attn_2 = nn.functional.interpolate(attn_2.clone(),u.shape[2:],mode="bilinear")
        attn =   attn_0#+attn_1#attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u
    
class SACCPA(nn.Module):
    def __init__(self, d_model,k_size=None): #ksize ignored
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = UAttention(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x



class SaccpaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(SaccpaBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.saccpa = SACCPA(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.saccpa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SaccpaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(SaccpaBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.saccpa = SACCPA(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.saccpa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3],initial_layer =3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(initial_layer, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def saccpa_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False,*args,**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(SaccpaBasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size,*args,**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def saccpa_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False,*args,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(SaccpaBasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size,*args,**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def saccpa_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False,*args,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing saccpa_resnet50......")
    model = ResNet(SaccpaBottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size,*args,**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def saccpa_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False,*args,**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SaccpaBottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size,*args,**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def saccpa_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False,*args,**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SaccpaBottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size,*args,**kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
