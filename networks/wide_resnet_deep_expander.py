import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import math
import sys
import numpy as np

def conv3x3_expand(in_planes, out_planes, expandsize, stride=1):
    "3x3 convolution with padding"
    return ExpanderConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, expandSize=(in_planes//expandsize))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

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
        #self.weight = conv_initm.weight
        self.fpWeight = torch.nn.Parameter(data=torch.Tensor(outdim, indim, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.fpWeight.data,mode='fan_out')

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
        # self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, dataInput):
        extendWeights = self.fpWeight.clone()
        extendWeights.mul_(self.mask.data)
        return torch.nn.functional.conv2d(dataInput, extendWeights, bias=None,
                                          stride=self.conStride, padding=self.conPad,
                                          dilation=self.conDil, groups=self.conGroups)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        #init.xavier_uniform_(gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, expandsize=2):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3_expand(in_planes, planes,
            expandsize=expandsize)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_expand(planes, planes, stride=stride,
            expandsize=expandsize)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_Deep_Expander(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_Deep_Expander, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet-Deep-Expander %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        # expandsize = 2

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*int(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet_Deep_Expander(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
