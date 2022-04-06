import torch
import torch.nn as nn
from models.deform.modules import DeformConv


class BasicChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.weighting = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=in_channels // reduction,
                                                 kernel_size=1,
                                                 padding=0,
                                                 stride=1, ),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=in_channels // reduction,
                                                 out_channels=in_channels,
                                                 kernel_size=1,
                                                 padding=0,
                                                 stride=1, ),
                                       nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avgpool(x)
        w = self.weighting(w)
        return x * w.expand_as(x)


class DCA(nn.Module):

    def __init__(self, in_channels, reduction, cda=False):
        super().__init__()

        self.cda = cda

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.weighting = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels // reduction,
                      kernel_size=1,
                      padding=0,
                      stride=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // reduction,
                      out_channels=in_channels,
                      kernel_size=1,
                      padding=0,
                      stride=1, )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w1 = self.avgpool(x)
        w2 = self.maxpool(x)

        w1 = self.weighting(w1)
        w2 = self.weighting(w2)

        w = self.sigmoid(w1 + w2)

        return x * w.expand_as(x)


class BasicSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3x3 = nn.Conv2d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=3,
                                 padding=1)

        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgmap = torch.mean(x, dim=1, keepdim=True)

        w = self.bn(self.conv3x3(avgmap))
        w = self.sigmoid(w)

        return x * w.expand_as(x)


class DSA(nn.Module):
    def __init__(self, cda=False):
        super().__init__()

        self.cda = cda

        self.offset = nn.Conv2d(in_channels=2,
                                out_channels=18,
                                kernel_size=3,
                                padding=1)
        self.conv3x3 = DeformConv(in_channels=2,
                                  out_channels=1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgmap = torch.mean(x, dim=1, keepdim=True)
        maxmap = torch.max(x, dim=1, keepdim=True)[0]

        catmap = torch.cat([maxmap, avgmap], dim=1)
        catmap = self.bn(self.conv3x3(catmap, self.offset(catmap)))
        w = self.sigmoid(catmap)

        return x * w


class CDA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 residual: bool):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dca = DCA(in_channels=in_channels,
                       reduction=16,
                       cda=True)
        self.dsa = DSA(cda=True)
        self.residual = residual

    def forward(self, x):
        y = x
        y = self.dca(y)
        y = self.dsa(y)

        if self.residual and self.in_channels == self.out_channels:
            y += x

        return y
