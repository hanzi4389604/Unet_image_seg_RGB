"""
*_* coding:utf-8 *_*
time:            2021/11/10 15:59
author:          丁治
remarks：        备注信息
"""
import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.transforms import transforms


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), padding_mode='reflect'),  # 镜像翻转填充方式
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),  # 防止过拟合，图像切割的数据一般都比较少，数据量太小，很容易过拟合。数据量大的时候不需要。
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), padding_mode='reflect'),  # 镜像翻转填充方式
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.4),  # 防止过拟合，图像切割的数据一般都比较少，数据量太小，很容易过拟合。数据量大的时候不需要
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSimpling(nn.Module):
    def __init__(self, c_in_out):
        super(DownSimpling, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in_out, c_in_out, (3, 3), stride=(2, 2), padding=(1, 1), padding_mode='reflect'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSimpling(nn.Module):
    def __init__(self, c_in_out):
        super(UpSimpling, self).__init__()
        self.C = nn.Conv2d(c_in_out, c_in_out//2, (1, 1))

    def forward(self, x, r):
        x = self.C(interpolate(x, scale_factor=2, mode='nearest'))  # 特征图上采样
        return torch.cat((x, r), 1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = CNNLayer(3, 64)
        self.c2 = nn.Sequential(
            DownSimpling(64),
            CNNLayer(64, 128),
        )
        self.c3 = nn.Sequential(
            DownSimpling(128),
            CNNLayer(128, 256),
        )
        self.c4 = nn.Sequential(
            DownSimpling(256),
            CNNLayer(256, 512),
        )
        self.c5 = nn.Sequential(
            DownSimpling(512),
            CNNLayer(512, 1024),
        )
        self.up4 = UpSimpling(1024)
        self.u4 = CNNLayer(512+1024//2, 512)
        self.up3 = UpSimpling(512)
        self.u3 = CNNLayer(256+512//2, 256)
        self.up2 = UpSimpling(256)
        self.u2 = CNNLayer(128+256//2, 128)
        self.up1 = UpSimpling(128)
        self.u1 = CNNLayer(64+128//2, 64)
        self.out = nn.Conv2d(64, 3, (3, 3), padding=(1, 1))
        self.S = nn.Sigmoid()  # 这里只做二分类

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(r1)
        r3 = self.c3(r2)
        r4 = self.c4(r3)
        x4 = self.c5(r4)
        x3 = self.u4(self.up4(x4, r4))
        x2 = self.u3(self.up3(x3, r3))
        x1 = self.u2(self.up2(x2, r2))
        hidden = self.u1(self.up1(x1, r1))
        return self.S(self.out(hidden))


if __name__ == '__main__':
    data = torch.randn(1, 3, 512, 512).cuda()
    # CNNLayer(3, 64)(data)
    print(UNet().cuda()(data).shape)

    # trans = transforms.ToTensor()
