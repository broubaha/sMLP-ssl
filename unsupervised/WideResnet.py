#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.nn import BatchNorm2d

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np
'''
    As in the paper, the wide resnet only considers the resnet of the pre-activated version, 
    and it only considers the basic blocks rather than the bottleneck blocks.
'''


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=0.001)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_chan, momentum=0.001)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.pre_res_act = pre_res_act
        # self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu1(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = act1 if self.pre_res_act else x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out = shortcut + residual
        return out

    def init_weight(self):
        # for _, md in self.named_modules():
        #     if isinstance(md, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #         if md.bias is not None:
        #             nn.init.constant_(md.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class WideResnetBackbone(nn.Module):
    def __init__(self, k=1, n=28, drop_rate=0):
        super(WideResnetBackbone, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(3)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1,
            drop_rate=drop_rate,
            pre_res_act=True,
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.init_weight()

    def create_layer(
            self,
            in_chan,
            out_chan,
            bnum,
            stride=1,
            drop_rate=0,
            pre_res_act=False,
    ):
        layers = [
            BasicBlockPreAct(
                in_chan,
                out_chan,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat)  # 1/2
        feat4 = self.layer3(feat2)  # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)
        return feat2, feat4

    def init_weight(self):
        # for _, child in self.named_children():
        #     if isinstance(child, nn.Conv2d):
        #         n = child.kernel_size[0] * child.kernel_size[0] * child.out_channels
        #         nn.init.normal_(child.weight, 0, 1. / ((0.5 * n) ** 0.5))
        #         #  nn.init.kaiming_normal_(
        #         #      child.weight, a=0.1, mode='fan_out',
        #         #      nonlinearity='leaky_relu'
        #         #  )
        #
        #         if child.bias is not None:
        #             nn.init.constant_(child.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class WideResnet(nn.Module):
    '''
    for wide-resnet-28-10, the definition should be WideResnet(n_classes, 10, 28)
    '''

    def __init__(self, n_classes, k=2, n=28, low_dim=64, proj=True):
        super(WideResnet, self).__init__()
        self.n_layers, self.k = n, k
        self.backbone = WideResnetBackbone(k=k, n=n)
        self.classifier = nn.Linear(64 * self.k, n_classes, bias=True)
        
        self.proj = proj
        if proj:
            self.l2norm = Normalize(2)

            self.fc1 = nn.Linear(64 * self.k, 64 * self.k)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(64 * self.k, low_dim)
      

    def forward(self, x):
        feat1 = self.backbone(x)[-1]
        feat = torch.mean(feat1, dim=(2, 3))
        out = self.classifier(feat)
        
        if self.proj:
            feat = self.fc1(feat)
            feat = self.relu_mlp(feat)       
            feat = self.fc2(feat)

            feat = self.l2norm(feat)   
            return out,feat
        else:
            return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.classifier.weight)
        if not self.classifier.bias is None:
            nn.init.constant_(self.classifier.bias, 0)

class WarmupCosineLrScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
            
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            
            ratio = np.cos((8 * np.pi * real_iter) / (16 * real_max_iter))   #utilisé         
            ##ratio = 0.5 * (1. + np.cos(np.pi * real_iter / real_max_iter))
            #ratio = np.exp(-2.77 * real_iter/real_max_iter)
            #ratio = np.exp(-2.77 * real_iter / real_max_iter) -0.062 * real_iter / real_max_iter
            # if real_iter>= real_max_iter//2 :
            #     ratio = ratio/2
            # if real_iter>= real_max_iter/3*2 :
            #     ratio = ratio/2
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


    def state_dict(self):
        # Save the state of the scheduler
        return {
            'warmup_epochs': self.warmup,
            'max_epochs': self.max_iter,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict):
        # Load the state of the scheduler
        self.warmup = state_dict['warmup_epochs']
        self.max_iter = state_dict['max_epochs']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']



if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    lb = torch.randint(0, 10, (2,)).long()

    net = WideResnetBackbone()
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    del net, out

    net = WideResnet(n_classes=10,proj=False)
    criteria = nn.CrossEntropyLoss()
    out = net(x)
    print(out.size())