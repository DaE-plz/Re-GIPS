import torch
import torch.nn as nn

import Geometry_transformation
import Image_Generator
import Image_discriminator
import geometry_encoder
import texture_encoder
import numpy as np



class DLGIPS_Geometry_encoder(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(DLGIPS_Geometry_encoder, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.geometry_encoder = geometry_encoder.Geometry_Encoder(self.input_channel, self.output_channel)

# projcetion_tensor (1,1,300,180) --->geometry_features(1,1,300,180),texture_features(1,1,300,180)
    def forward(self, projcetion_tensor):
        geometry_features = self.geometry_encoder(projcetion_tensor)
        return geometry_features

class DLGIPS_Texture_encoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DLGIPS_Texture_encoder, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.texture_encoder = texture_encoder.Texture_Encoder(self.input_channel, self.output_channel)

    # projcetion_tensor (1,1,300,180) --->geometry_features(1,1,300,180),texture_features(1,1,300,180)
    def forward(self, projcetion_tensor):
        texture_features = self.texture_encoder(projcetion_tensor)
        return texture_features


class DLGIPS_geometry_transform(nn.Module):
    def __init__(self, input_channel,image_size, proj_size,
                 num_proj_initialization,start_angle_initialization,end_angle_initialization,
                 num_proj_transform,start_angle_transform,end_angle_transform):
        super(DLGIPS_geometry_transform, self).__init__()
        self.input_channel = input_channel
        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj_initialization = num_proj_initialization
        self.start_angle_initialization = start_angle_initialization
        self.end_angle_initialization = end_angle_initialization
        self.num_proj_transform = num_proj_transform
        self.start_angle_transform = start_angle_transform
        self.end_angle_transform = end_angle_transform

        self.geometry_transformation = Geometry_transformation.GeometryTransformation( self.input_channel,
                                                                                       self.image_size, self.proj_size,
                                                                                       self.num_proj_initialization,self.start_angle_initialization,self.end_angle_initialization,
                                                                                       self.num_proj_transform,self.start_angle_transform,self.end_angle_transform)
# geometry_features (1,1,300,180)  ---> transformed_features (1,2,300,180)
    def forward(self, geometry_features):
        transformed_features = self.geometry_transformation(geometry_features)
        transformed_features_ap, transformed_features_lt = torch.chunk(transformed_features, 2, dim=1)

        return transformed_features_ap, transformed_features_lt

class DLGIPS_image_generator(nn.Module):
    def __init__(self, geometry_channels, texture_channels):
        super(DLGIPS_image_generator, self).__init__()
        self.geometry_channels = geometry_channels
        self.texture_channels = texture_channels

        #self.image_generator = Image_Generator.ImageGenerator(self.geometry_channels,self.texture_channels)
        self.image_generator = Image_Generator.UNet(self.geometry_channels,self.texture_channels)


# 在TASK1中，transformed_geometry_features （1，2，300，180）texture_features （1，1，300，180）
# transformed_features_ap （1，1，300，180）, transformed_features_lt （1，1，300，180）
    def forward(self, transformed_geometry_features, texture_features):

        generated_image = self.image_generator(transformed_geometry_features, texture_features)
        return generated_image

        # elif transformed_geometry_features.shape == (1, 4, 300, 180):
        #     chunks = torch.chunk(transformed_geometry_features, 4, dim=1)
        #     generated_images = [self.image_generator(chunk, texture_features) for chunk in chunks]
        #     generated_image_ap, generated_image_30, generated_image_60, generated_image_lt = generated_images
        #     return generated_image_ap,generated_image_30,generated_image_60,generated_image_lt

class DLGIPS_image_discriminator(nn.Module):
    def __init__(self):
        super(DLGIPS_image_discriminator, self).__init__()

        self.image_discriminator = Image_discriminator.ImageDiscriminator()

    def forward(self, generated_image):
        out_scale1, out_scale2, out_scale3 = self.image_discriminator(generated_image)

        return out_scale1, out_scale2, out_scale3
"""

    def forward(self, generated_image_ap,generated_image_30,generated_image_60,generated_image_lt):
        score_ap = self.image_discriminator(generated_image_ap)
        score_30 = self.image_discriminator(generated_image_30)
        score_60 = self.image_discriminator(generated_image_60)
        score_lt = self.image_discriminator(generated_image_lt)
        return score_ap,score_30,score_60,score_lt
"""
# input_channel=1
# output_channel=1
#
# image_size= [500, 128, 128]
# proj_size= [300, 180]
# num_proj=1
# start_angle = - np.pi / 4
# end_angle = (3 * np.pi) / 4
# geometry_encoder = Feature_encoder.FeatureEncoder(input_channel, output_channel, True)
# texture_encoder = Feature_encoder.FeatureEncoder(input_channel, output_channel, False)
# geo_transform=Geometry_transformation.GeometryTransformation( input_channel,image_size, proj_size, num_proj,start_angle,end_angle)
# image_generator = Image_Generator.ImageGenerator(input_channel ,output_channel)
# image_discriminator = Image_discriminator.ImageDiscriminator(input_channel)
#
# # train AP_AP&LT
# # 导入 AP角度的投影 dataloader
# root_dir = 'G:/dataset/LIDC-IDRI'
# target_shape = (500, 128, 128)
# proj_size = [300, 180]
# start_angle = - np.pi / 4
# end_angle = (3 * np.pi) / 4
# num_proj=2
# # projection_datasets (1,2,300,180) 两个角度
# projection_datasets = projection_dataset.ProjectionDataset(root_dir, target_shape, proj_size,num_proj, start_angle, end_angle)
#
# projection_dataloader = projection_dataset.DataLoader(projection_datasets, batch_size=1, shuffle=True)
#
# device = torch.device("cuda")
# batch_size = 1
# num_epochs=3
# for epoch in range(num_epochs):
#     for data in projection_dataloader:
#         projection_data_gpu = data.to(device)
#         geometry=geometry_encoder(projection_data_gpu)
#         texture=texture_encoder(projection_data_gpu)
#         geometry_after_transform=geo_transform(geometry)