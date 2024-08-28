from __future__ import print_function

import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

__all__ = ['alexnet_gn']

class alexnet_gn(nn.Module):
    def __init__(self, in_channel=3, out_dim=128, l2norm=True):
        super(alexnet_gn, self).__init__()
        self._l2norm = l2norm
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, 11, 4, 2, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.ave_pool = nn.AdaptiveAvgPool2d((6,6))
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, out_dim)
        )
        if self._l2norm: self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.ave_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        if self._l2norm: x = self.l2norm(x)
        return x

    def compute_feat(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        if self._l2norm: x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':

    import torch
    model = alexnet_gn().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()
    out = model.compute_feat(data, 5)

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)