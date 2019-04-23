# https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py
# https://github.com/drimpossible/Deep-Expander-Networks/blob/master/code/models/densenetexpander_cifar.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import sys
import math
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_expand(in_planes, out_planes, expandsize, stride=1):
    "3x3 convolution with padding"
    return ExpanderConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, expandSize=(in_planes//expandsize))

class ExpanderConv2d(nn.Module):
    def __init__(self, indim, outdim, kernel_size, expandSize,
                 stride=1, padding=0, inDil=1, groups=1, mode='random'):

        super(ExpanderConv2d, self).__init__()
        # Initialize all parameters that the convolution function needs to know
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conGroups = groups
        #self.weight = 5
        self.bias = True
        self.weight = torch.nn.Parameter(data=torch.Tensor(outdim, indim, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.weight.data,mode='fan_out')

        self.mask = torch.zeros(outdim, (indim),1,1)

        if indim > outdim:
            for i in range(outdim):
                x = torch.randperm(indim)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(indim):
                x = torch.randperm(outdim)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1

        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, dataInput):
        extendWeights = self.weight.clone()
        extendWeights.mul_(self.mask.data)
        return torch.nn.functional.conv2d(dataInput, extendWeights, bias=None,
                                          stride=self.conStride, padding=self.conPad,
                                          dilation=self.conDil, groups=self.conGroups)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        # init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

# Be Here Now 

class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, expandsize=2):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ExpanderConv2d(in_planes, planes, kernel_size=1, expandSize=((planes)//expandsize))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ExpanderConv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, expandSize=((planes)//expandsize))
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = ExpanderConv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, 
                                expandSize=((planes*1)//expandsize))
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class Pyramid_Deep_Expand(nn.Module):
    def __init__(self, depth, num_classes, bottleneck=True, alpha=48, expandsize=2):
        
        super(Pyramid_Deep_Expand, self).__init__()
        
        self.inplanes = 16

        # block, num_blocks = cfg(depth)

        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            # block = Bottleneck

        self.addrate = alpha / (3*n*1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = ExpanderConv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, expandSize=(3//expandsize))
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    net=Pyramid_Deep_Expand(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
