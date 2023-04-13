import torch
import torch.nn as nn

from mmcv.ops import DeformConv2d as DeformConv


class SCA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.convpool = nn.Conv2d(in_channels, in_channels, 4, stride=4, groups=in_channels)
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 4, 4)
        self.channel_connection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor):
        maxpool = self.maxpool(x)
        convpool = self.convpool(x)
        pools = maxpool + convpool
        pools = self.channel_connection(pools)
        filled_pools = self.deconv(pools)
        filled_pools = self.sigmoid(filled_pools)
        return x * filled_pools

    def __repr__(self):
        return "Spatial-aware Channel Attention"


class CDSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int) -> None:
        super().__init__()
        self.reduction = reduction
        self.bn = nn.BatchNorm2d(in_channels // reduction, eps=1e-5, momentum=0.1, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.pwconv1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels // reduction,
                                 kernel_size=1)
        self.pwconv2 = nn.Conv2d(in_channels=in_channels // reduction,
                                 out_channels=in_channels,
                                 kernel_size=1)
        self.offset = nn.Conv2d(in_channels=in_channels // reduction,
                                out_channels=18,
                                kernel_size=3,
                                padding=1)
        self.conv3x3 = DeformConv(in_channels=in_channels // reduction,
                                  out_channels=in_channels // reduction,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        w = self.pwconv1(x)
        w = self.conv3x3(w, self.offset(w))
        w = self.bn(w)
        w = self.pwconv2(w)
        w = self.sigmoid(w)
        y = x * w
        return y

    def __repr__(self):
        return "Channel-modulated Deformable Spatial Attention"


class DCA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction_s: int,
                 reduction_c: int) -> None:
        super().__init__()
        self.sca = SCA(in_channels, reduction_c)
        self.cdsa = CDSA(in_channels, reduction_s)

    def forward(self, x):
        y = self.sca(x)
        z = self.cdsa(y)
        return z

    def __repr__(self):
        return "Dual-dimension Combined Attention"


class AttentionAtFusion(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size,
                 reduction=4):
        super().__init__()
        self.channel = channel
        self.channel_connection = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )

    def forward(self,
                s: torch.Tensor,
                d: torch.Tensor = None):
        sw = self.channel_connection(s)
        d = d * sw
        merge = s + d
        return merge

    def __repr__(self):
        return "Attention at fusion"
