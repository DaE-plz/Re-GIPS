import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch
import ct_geometry_projector



"""
输入是ap方向的投影，调用的是AP_AP.backward，从而得到volume
得到的volume经过refinment model,进行补全后
调用的是AP_APLT.forward,从而得到ap和lt方向的projcetion
"""



#  1 --> 1
class AP_AP_Operators():
    def __init__(self, image_size, proj_size, num_proj, start_angle, end_angle):

        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.raw_reso = 2.8

        # Initialize required parameters for image, view, detector
        geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
                                                                  num_proj=self.num_proj,
                                                                  start_angle=self.start_angle,
                                                                  end_angle=self.end_angle,
                                                                  proj_size=self.proj_size,
                                                                  raw_reso=self.raw_reso)

        # Initialize forward and backward projectors for both AP and LT.
        self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
        self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)
    def backward_project(self, projection_data_ap):
        """
        Backward Projection Operator:
        projection_data_ap  ：tensor  (1,1,300,180)
        要先转化成 （1，1，1，300，180）的tensor
         return : (1,1,1500,128,128)
        """
        expanded_tensor = projection_data_ap.unsqueeze(1)
        volume = self.backward_projector(expanded_tensor)
        # print(f" Tensor AP_AP Backward  3D volume shape: {volume.shape}")
        return volume

    def forward_project(self, volume):
        """
        Forward Projection for AP-LT Transformation:
        Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
        输入的volume是（1，1，500，128，128）
        得到的 proj 是（1，1，1，300，180）
        然后将 得到的 proj进行 去除维度，变成 （1，1，300，180），方便进入后续的神经网络
        """
        projections_ap_ap = self.forward_projector(volume)
        reduced_tensor = projections_ap_ap.squeeze(1)
        print(f" Tensor Volume_Forward projection AP shape: {reduced_tensor.shape}")
        return reduced_tensor

#  1 --> 2
class AP_APLT_Operators():
    def __init__(self, image_size, proj_size, num_proj,start_angle,end_angle):

        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle=start_angle
        self.end_angle=end_angle
        self.raw_reso = 2.8

        # Initialize required parameters for image, view, detector
        geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
                                            num_proj=self.num_proj,
                                            start_angle=self.start_angle,
                                            end_angle=self.end_angle,
                                            proj_size=self.proj_size,
                                            raw_reso=self.raw_reso)


        # Initialize forward and backward projectors for both AP and LT.
        self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
        self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)

    def backward_project(self, projection_data_ap):
        """
        Backward Projection Operator:
        projection_data_ap  ：tensor  (1,1,300,180)
        要先转化成 （1，1，1，300，180）的tensor
        得到的
              """
        expanded_tensor = projection_data_ap.unsqueeze(1)
        volume = self.backward_projector(expanded_tensor)
        # print(f" Tensor AP_AP&LT Backward  3D volume shape: {volume.shape}")
        return volume


    def forward_project(self, volume):
        """
        Forward Projection for AP-LT Transformation:
        Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.
        输入的volume是（1，1，500，128，128）
        得到的 proj 是（1，1，2，300，180）
        然后将 得到的 proj进行 去除维度，变成 （1，2，300，180），方便进入后续的神经网络
        """
        projections_ap_ap=self.forward_projector(volume)
        reduced_tensor = projections_ap_ap.squeeze(1)
        # print(f" Tensor Volume_Forward projection AP&LT shape: {reduced_tensor.shape}")
        return reduced_tensor

#     #AP-AP
#     start_angle1 = - np.pi / 2
#     end_angle1 = np.pi / 2
#     # AP-AP&LT
#     num_proj2 = 2
#     start_angle2 = - np.pi / 4
#     end_angle2 = ( 3 * np.pi ) / 4
#

# if __name__ == '__main__':
#     # 验证forward
#     '''
#     验证 AP-AP
#     '''
#
#     # # 初始化
#     target_shape = [500, 128, 128]
#     proj_size = [300, 180]
#     # num_proj1 = 1
#     # start_angle1 = (- np.pi) /2
#     # end_angle1 = np.pi /2
#     # ap_ap_operators = AP_AP_Operators(target_shape, proj_size, num_proj1, start_angle1, end_angle1)
#     #
#     # 导入 volume
#     dicom_folder = 'G:/dataset_01/LIDC-IDRI/LIDC-IDRI-0001'
#     volume=volume_padding.symmetric_padding_3d_volume(dicom_folder,target_shape)  #(500,128,128)
#     tensor=torch.from_numpy(volume)
#     volume_tensor = tensor.unsqueeze(0).unsqueeze(0)  #(1,1,500,128,128)
#     #
#     # # 验证 forward projection
#     # projections_ap_=ap_ap_operators.forward_project(volume_tensor) # (1,1,300,180)
#     # projections_ap_reduced_tensor = projections_ap_.squeeze(0).squeeze(0)  #(300,180)
#     #
#     # # 绘制AP投影
#     # plt.imshow(projections_ap_reduced_tensor, cmap='gray')
#     # plt.title('ap_ap Projection')
#     # plt.show()
#
#     '''
#     验证 AP-AP & LT
#     '''
#     # 初始化
#     num_proj2 = 2
#     start_angle2 = - np.pi / 2
#     end_angle2 =  np.pi
#     ap_aplt_operators = AP_APLT_Operators(target_shape, proj_size, num_proj2, start_angle2, end_angle2)
#     # 验证 forward projection
#     projections = ap_aplt_operators.forward_project(volume_tensor) #（1，2，300，180）
#     # projection_reduce=projections.squeeze(0) #（2，300，180）
#     numpy_array = projections.numpy()
#     projections_ap_ap=numpy_array[0, 0, :, :]
#     projections_ap_lt=numpy_array[0, 1, :, :]
#     #
#     #
#     # 设置绘图
#     plt.figure(figsize=(12, 6))
#
#     # 绘制AP投影
#     plt.subplot(1, 2, 1)  # 1行2列的第1个
#     plt.imshow(projections_ap_ap, cmap='gray')
#     plt.title('ap_ap Projection')
#
#     # 绘制LT投影
#     plt.subplot(1, 2, 2)  # 1行2列的第2个
#     plt.imshow(projections_ap_lt, cmap='gray')
#     plt.title('ap_lt Projection')
#
#     # 显示图像
#     plt.show()
#
# #     projection_data_np_0 = np.load('projection_0.npy')
# #     # projection_0 shape (300,180)
# #     proj_tensor = torch.from_numpy(projection_data_np_0)
# #     proj_0_tensor = proj_tensor.unsqueeze(0).unsqueeze(0)
# #     volume_ap_aplt = ap_ap_operators.backward_project(proj_0_tensor)
