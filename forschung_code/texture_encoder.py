import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return F.relu(out)


class Texture_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Texture_Encoder, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.InstanceNorm2d(16)
        self.downsample1 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(32)
        self.downsample2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(64)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])

        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm_up1 = nn.InstanceNorm2d(32)
        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.norm_up2 = nn.InstanceNorm2d(16)
        self.final_up_conv = nn.Conv2d(16, out_channels, kernel_size=7, stride=1, padding=3)
        self.norm_up3 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.norm1(self.initial_conv(x)))
        x = F.relu(self.norm2(self.downsample1(x)))
        x = F.relu(self.norm3(self.downsample2(x)))
        x = self.res_blocks(x)
        x = F.relu(self.norm_up1(self.upsample1(x)))
        x = F.relu(self.norm_up2(self.upsample2(x)))
        x = F.relu(self.norm_up3(self.final_up_conv(x)))
        return x

  # x:ap角度的投影（1，1，300，180） 得到的结果是 texture_feature (1,1,300,180)

if __name__ == '__main__':
    x=torch.randn(1,1,300,180)
    net= Texture_Encoder(1,1)
    print(net(x).shape)