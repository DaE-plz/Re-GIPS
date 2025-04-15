
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.scale1 = DiscriminatorScale()
        self.scale2 = DiscriminatorScale()
        self.scale3 = DiscriminatorScale()
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out_scale1 = self.scale1(x)
        x_downsampled = self.avg_pool(x)
        out_scale2 = self.scale2(x_downsampled)
        x_downsampled = self.avg_pool(x_downsampled)
        out_scale3 = self.scale3(x_downsampled)
        return out_scale1, out_scale2, out_scale3

class DiscriminatorScale(nn.Module):
    def __init__(self):
        super(DiscriminatorScale, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    net=ImageDiscriminator()
    x=torch.randn(1,1,300,180)


    # 将输入张量传递给 discriminator 并获取输出
    output_scale1, output_scale2, output_scale3 = net(x)

    # 打印输出形状，以验证尺寸是否正确
    print(f"Output scale 1 shape: {output_scale1.shape}")
    print(f"Output scale 2 shape: {output_scale2.shape}")
    print(f"Output scale 3 shape: {output_scale3.shape}")