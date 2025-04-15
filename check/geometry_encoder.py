import torch
import torch.nn as nn
import torch.nn.functional as F
import projection_dataset
from torch.utils.data import DataLoader

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

class Geometry_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Geometry_Encoder, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, padding=3)
        self.downsample = nn.Sequential(
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.final_layers(x)
        return x
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.norm1 = nn.InstanceNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.norm2 = nn.InstanceNorm2d(channels)
#
#     def forward(self, x):
#         identity = x
#         out = F.relu(self.norm1(self.conv1(x)))
#         out = self.norm2(self.conv2(out))
#         out += identity
#         return F.relu(out)
#
#
# class Geometry_Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Geometry_Encoder, self).__init__()
#         self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=7, stride=1, padding=3)
#         self.norm1 = nn.InstanceNorm2d(16)
#         self.downsample1 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
#         self.norm2 = nn.InstanceNorm2d(32)
#         self.downsample2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.norm3 = nn.InstanceNorm2d(64)
#         self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
#
#         self.final_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.norm5 = nn.InstanceNorm2d(64)
#
#         self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.norm_up1 = nn.InstanceNorm2d(32)
#         self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
#         self.norm_up2 = nn.InstanceNorm2d(16)
#         self.final_up_conv = nn.Conv2d(16, out_channels, kernel_size=7, stride=1, padding=3)
#         self.norm_up3 = nn.InstanceNorm2d(out_channels)
#
#
#         self.final_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.norm4 = nn.InstanceNorm2d(32)  # ()
#
#
#         # else:
#         #     self.final_conv = None
#
#     def forward(self, x):
#         x = F.relu(self.norm1(self.initial_conv(x)))
#         x = F.relu(self.norm2(self.downsample1(x)))
#         x = F.relu(self.norm3(self.downsample2(x)))
#         x = self.res_blocks(x)
#        # xx
#         x = F.relu(self.norm4(self.final_conv(x)))
#         x = F.relu(self.norm5(self.final_conv2(x)))
#
#         x = F.relu(self.norm_up1(self.upsample1(x)))
#         x = F.relu(self.norm_up2(self.upsample2(x)))
#         x = F.relu(self.norm_up3(self.final_up_conv(x)))
#         return x
#   # x:ap角度的投影（1，1，300，180） 得到的结果是 geometry_feature (1,1,300,180)

if __name__ == '__main__':
    ap_dir = 'G:/dataset/projection_ap'
    lt_dir = 'G:/dataset/projection_lt'

    # Initialize the dataset and DataLoader
    dataset = projection_dataset.ProjectionTensorDataset(ap_dir, lt_dir)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Initialize the Geometry_Encoder
    in_channels = 1  # This should be set according to your input's number of channels
    out_channels = 1  # This should be set according to your desired output's number of channels
    geometry_encoder = Geometry_Encoder(in_channels, out_channels)

    # Move the geometry encoder to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry_encoder = geometry_encoder.to(device)

    # Process the data
    for i, (ap_projection, lt_projection) in enumerate(dataloader):
        # We will only check the first batch
        if i == 0:
            # Move the data to the same device as the model
            ap_projection = ap_projection.to(device)

            # Forward pass through the geometry encoder
            output = geometry_encoder(ap_projection)

            # Check the output shape
            print(f"Output shape after geometry encoder: {output.shape}")

            # If you want to inspect the values, you need to move the tensor back to CPU
            output = output.to('cpu').detach().numpy()
            print(f"Output values: {output}")

            # In this case, we break after the first batch, but you can remove the break
            # statement if you want to process the entire dataset
            break