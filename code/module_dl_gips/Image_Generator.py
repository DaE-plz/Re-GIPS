import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGenerator(nn.Module):
    def __init__(self, geometry_channels, texture_channels):
        super(ImageGenerator, self).__init__()
        # 将几何特征和纹理特征在通道维度上拼接
        total_channels = geometry_channels + texture_channels  #2

        # 添加两个下采样层
        self.downsampling_blocks = nn.Sequential(
            DownsamplingBlock(total_channels, total_channels * 2),  # (2,4)
            DownsamplingBlock(total_channels * 2, total_channels * 4) # (4,8)
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(total_channels * 4) for _ in range(4)])
        self.upsampling_blocks = nn.ModuleList([
            UpsamplingBlock(total_channels * 4 , total_channels * 2),
            UpsamplingBlock(total_channels * 2, total_channels ),

        ])
        # 根据我们的图像信息，最后的卷积层应该输出 int16 类型的图像
        self.final_conv = nn.Conv2d(total_channels , 1, kernel_size=7, padding=3)

    def forward(self, geometry_features, texture_features):
        # 拼接几何特征和纹理特征
        x = torch.cat([geometry_features, texture_features], dim=1)
        # 应用下采样层
        x = self.downsampling_blocks(x)
        # 通过残差块
        x = self.res_blocks(x)
        # 应用上采样层
        for upsampling_block in self.upsampling_blocks:
            x = upsampling_block(x)
        # 应用最后的卷积层并使用tanh激活函数
        # x = torch.tanh(self.final_conv(x))
        x = F.relu(self.final_conv(x))
        # # 将输出缩放为 int16 类型
        # x = (x * 32767).type(torch.int16)
        return x

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

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return F.relu(x)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        # 第一个卷积层，带步长，用于减小特征图的尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # 第二个卷积层，用于进一步处理特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


# 在TASK1中，transformed_geometry_features （1，2，300，180）texture_features （1，1，300，180）
# transformed_features_ap （1，1，300，180）, transformed_features_lt （1，1，300，180）

if __name__ == '__main__':
    x=torch.randn(1,1,300,180)
    y=torch.randn(1,1,300,180)
    net= ImageGenerator(1,1)
    print(net(x,y).shape)