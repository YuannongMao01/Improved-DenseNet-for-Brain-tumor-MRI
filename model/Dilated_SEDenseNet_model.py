# Dilated SE-model
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


class SEDenseBottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=32, dropRate=0, kernel_size=3, dilation=1):
        super(SEDenseBottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # padding = dilation if dilation > 1 else 1
        padding = ((kernel_size - 1) // 2) * dilation
        
        
        # Depthwise Separable Convolution
        self.depthwise = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                                   padding=padding, groups=planes, bias=False, dilation=dilation)

        self.pointwise = nn.Conv2d(planes, growthRate, kernel_size=1, bias=False)


        outplanes = inplanes + growthRate

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(outplanes, outplanes // 16)
        self.fc2 = nn.Linear(outplanes // 16, outplanes)
        
        self.logSigmoid = nn.LogSigmoid()
        # self.softplus =  nn.Softplus()
        # self.sigmoid =  nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)

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
        se = self.logSigmoid(se)
        # se = self.relu(se)
        se = se.view(se.size(0), se.size(1), 1, 1)
        out = out * se.expand_as(out)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avgpool(out)
        return out


class SE_DenseNet(nn.Module):
    def __init__(self, growthRate=32, LK_head=True, dropRate=0,
                 increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16), num_classes=4, kernel_sizes = [3, 3, 3, 3], dilation_layers = [1, 1, 2, 3]):
        super(SE_DenseNet, self).__init__()

        block = SEDenseBottleneck
        self.growthRate = growthRate
        self.dropRate = dropRate
        self.increasingRate = increasingRate
        
        headplanes = growthRate * pow(increasingRate, 2)
        self.inplanes = headplanes * 2  # default 64

        self.LK_head = LK_head # large kernel head
        if self.LK_head:
            self.conv1 = nn.Conv2d(3, headplanes * 2, kernel_size=7, stride=2, padding=3, bias=False)
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

        # Dense-Block 1 and transition (56x56)
        dilation_layers
        self.dense1 = self._make_layer(block, layers[0], kernel_sizes[0], block_num=1, dilation = dilation_layers[0])
        self.trans1 = self._make_transition(compressionRate)
        # Dense-Block 2 and transition (28x28)
        self.dense2 = self._make_layer(block, layers[1], kernel_sizes[1], block_num=2, dilation = dilation_layers[1])
        self.trans2 = self._make_transition(compressionRate)
        # Dense-Block 3 and transition (14x14)
        self.dense3 = self._make_layer(block, layers[2],kernel_sizes[2], block_num=3, dilation = dilation_layers[2])
        self.trans3 = self._make_transition(compressionRate)
        # Dense-Block 4 (14x14)
        self.dense4 = self._make_layer(block, layers[3],kernel_sizes[3], block_num=4, dilation = dilation_layers[3])

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

    def _make_layer(self, block, blocks, kernel_size, block_num, dilation):
        layers = []
        for i, _ in enumerate(range(blocks)):
            # dilation = 1
            # if block_num == 2:
            #     dilation = 2
            # if block_num == 3:
            #     dilation = 5
            # elif block_num == 4:
                # dilation = 9
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate, kernel_size=kernel_size, dilation=dilation))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        self.growthRate *= self.increasingRate
        return Transition(inplanes, outplanes)

    def forward(self, x):
        if self.LK_head:
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

