# Dilated SE-model

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


class SEDenseBottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=32, dropRate=0, dilation = 1):
        super(SEDenseBottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        padding = dilation

        # Depthwise Separable Convolution
        self.depthwise = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=padding, groups=planes, bias=False, dilation=dilation)

        self.pointwise = nn.Conv2d(planes, growthRate, kernel_size=1, bias=False)


        outplanes = inplanes + growthRate

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(outplanes, outplanes // 16)
        self.fc2 = nn.Linear(outplanes // 16, outplanes)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.conv2(out)

        out = self.depthwise(out)
        out = self.pointwise(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        se = self.global_avg(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)

        out = out * se.expand_as(out)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,
                               bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avgpool(out)
        return out


class SE_DenseNet(nn.Module):
    def __init__(self, growthRate=32, head7x7=True, dropRate=0,
                 increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16), num_classes=4):
        """ Constructor
        Args:
            layers: config of layers, e.g., (6, 12, 24, 16)
            num_classes: number of classes
        """
        super(SE_DenseNet, self).__init__()

        block = SEDenseBottleneck
        self.growthRate = growthRate
        self.dropRate = dropRate
        self.increasingRate = increasingRate
        headplanes = growthRate * pow(increasingRate, 2)
        self.inplanes = headplanes * 2  # default 64

        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, headplanes * 2, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(headplanes * 2)
        else:
            self.conv1 = nn.Conv2d(3, headplanes, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(headplanes)
            self.conv2 = nn.Conv2d(headplanes, headplanes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(headplanes)
            self.conv3 = nn.Conv2d(headplanes, headplanes * 2, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(headplanes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense-Block 1 
        # self.dense1 = self._make_layer(block, layers[0])
        self.dense1 = self._make_layer(block, layers[0], block_num=1)
        self.trans1 = self._make_transition(compressionRate)
        # Dense-Block 2 
        self.dense2 = self._make_layer(block, layers[1], block_num=2)
        self.trans2 = self._make_transition(compressionRate)
        # Dense-Block 3
        self.dense3 = self._make_layer(block, layers[2], block_num=3)
        self.trans3 = self._make_transition(compressionRate)
        # Dense-Block 4
        self.dense4 = self._make_layer(block, layers[3], block_num=4)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, block_num):
        layers = []
        for _ in range(blocks):
            dilation = 1
            if block_num == 3:  # Dense-Block 3
                dilation = 2
            elif block_num == 4:  # Dense-Block 4
                dilation = 4

            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate, dilation = dilation))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        self.growthRate *= self.increasingRate
        return Transition(inplanes, outplanes)

    def forward(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = self.bn(x)
        ## change final fully connect to Global Average Pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
