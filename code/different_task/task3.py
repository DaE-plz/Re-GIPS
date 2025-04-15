import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch
import ct_geometry_projector
import loaddcm
import matplotlib.pyplot as plt
import preprocessing

#  2 --> 4
class APLT_4_Operators():
    def __init__(self, image_size, proj_size, num_proj,start_angle,end_angle):
        # Initialize parameters and geometry for LT-LT conversion.
        self.image_size = image_size
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.end_angle=end_angle
        self.start_angle=start_angle
        self.raw_reso = 2.8
        geo_param = ct_geometry_projector.Initialization_ConeBeam(image_size=self.image_size,
                                            num_proj=self.num_proj,
                                            start_angle=self.start_angle,
                                            end_angle=self.end_angle,
                                            proj_size=self.proj_size,
                                            raw_reso=self.raw_reso)
        # Initialize forward and backward projectors for both AP and LT.
        self.forward_projector = ct_geometry_projector.Projection_ConeBeam(geo_param)
        self.backward_projector = ct_geometry_projector.FBP_ConeBeam(geo_param)

    def backward_project(self, projection_data_ap, projection_data_lt):
        """
        Backward Projection Operator:
        Takes 2D projection data and reconstructs the 3D volume.

        Arguments:
        projection_data: torch tensor with input size (B, num_proj, proj_size_h, proj_size_w)
        """
        # Perform the backward projection using the filtered back projection operator.
        # 创建一个形状为 (2, 300, 180) 的零张量
        projection_data_padded = np.zeros((self.num_proj, self.proj_size[0], self.proj_size[1]))

        # 将您的单个投影数据放置在第一个位置
        projection_data_padded[0, :, :] = projection_data_ap
        projection_data_padded[1, :, :] = projection_data_lt
        # 将 numpy 数组转换为 PyTorch 张量，并添加批次维度
        projection_data_tensor = torch.from_numpy(projection_data_padded).float().unsqueeze(0)
        print("1vs2 projection_data_tensor", projection_data_tensor.shape)

        volume = self.backward_projector(projection_data_tensor)

        # 将 PyTorch 张量转换为 NumPy 数组
        # 确保张量在 CPU 上
        volume_np = volume.cpu().detach().numpy()
        # 移除批次维度，只保留 3D 体积的空间维度
        volume_np_squeezed = volume_np[0]
        print(f" 1 vs 2 3D volume shape: {volume_np_squeezed.shape}")  # 应该输出 (133, 512, 512)

        # 保存为 TIFF 文件
        loaddcm.save_3d_volume_as_tif(volume_np_squeezed, 'lt_ltap_reconstructed_volume.tif')
        return volume_np_squeezed

    def forward_project(self, volume):
        """
        Forward Projection for AP-LT Transformation:
        Computes 2D projections at 0 degrees (AP) and 90 degrees (LT) from a 3D volume.

        Arguments:
        volume: torch tensor with input size (B, C, img_x, img_y, img_z)
        """

        volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
        projections = self.forward_projector(volume_tensor)

        # 转换为 NumPy 数组，并确保张量在 CPU 上
        projections_np = projections.cpu().detach().numpy()

        # 移除不需要的维度，只保留 (1, 300, 180)
        projections_np_squeezed = projections_np[0, 0]
        print(f"Reduced projections 0 shape: {projections_np_squeezed.shape}")  # 应该输出 (2, 300, 180)

        projections_aplt_ap = projections_np_squeezed[0]
        projections_aplt_30 = projections_np_squeezed[1]
        projections_aplt_60 = projections_np_squeezed[2]
        projections_aplt_lt = projections_np_squeezed[3]
        return projections_aplt_ap, projections_aplt_30,projections_aplt_60,projections_aplt_lt




if __name__ == '__main__':
    # 验证forward
    '''
    验证 AP-AP
    '''

    # 初始化
    image_size = [332, 128, 128]
    proj_size = [300, 180]
    num_proj = 4
    start_angle = - np.pi / 6
    end_angle = ( 2 * np.pi ) / 3
    aplt_4_operators = APLT_4_Operators(image_size, proj_size, num_proj,start_angle,end_angle)

    # 导入 volume
    dicom_folder = 'LIDC-IDRI-0001'
    volume,_,_,_ = preprocessing.preprocess_dicom_images(dicom_folder)

    # 验证 forward projection
    projections_aplt_ap ,projections_aplt_30 ,projections_aplt_60 ,projections_aplt_lt =aplt_4_operators.forward_project(volume)



    # 设置绘图
    plt.figure(figsize=(12, 6))

    # 绘制AP投影
    plt.subplot(1, 4, 1)  # 1行2列的第1个
    plt.imshow(projections_aplt_ap, cmap='gray')
    plt.title('APLT_AP Projection')

    # 绘制30投影
    plt.subplot(1, 4, 2)  # 1行2列的第1个
    plt.imshow(projections_aplt_30, cmap='gray')
    plt.title('APLT_30 Projection')

    # 绘制60投影
    plt.subplot(1, 4, 3)  # 1行2列的第1个
    plt.imshow(projections_aplt_60, cmap='gray')
    plt.title('APLT_60 Projection')

    # 绘制LT投影
    plt.subplot(1, 4, 4)  # 1行2列的第2个
    plt.imshow(projections_aplt_lt, cmap='gray')
    plt.title('APLT_LT Projection')

    # 显示图像
    plt.show()

    # # 验证 backward
    # '''
    # 验证 AP-AP
    # '''
    #
    # # 加载测试数据
    # projection_data_np_90 = np.load('projection_90.npy')
    # # projection_90 shape (300,180)
    #
    # volume_lt = lt_lt_operators.backward_project(projection_data_np_90)
    #
    #
    # '''
    # 验证 AP-AP & LT
    # '''
    # # 加载测试数据
    # projection_data_np_0 = np.load('projection_0.npy')
    # # projection_90 shape (300,180)
    #
    # volume_lt_ap = lt_ltap_operators.backward_project(projection_data_np_0,projection_data_np_90)
    #
    # '''
    # 用单个LT的投影，backward得到的volume，去forward 验证 LT-LT
    # '''
    # # 验证 forward projection
    # single_single_projections_ap = lt_lt_operators.forward_project(volume_lt)[0]
    #
    # # 绘制AP投影
    # plt.imshow(single_single_projections_ap, cmap='gray')
    # plt.title('single projection LT_LT Projection')
    # plt.show()
    # '''
    # 用AP和LT的投影，backward得到的volume，去forward 验证 LT-AP&LT
    # '''
    # # 验证 forward projection
    # single_duo_projections_ap , single_duo_projections_lt= lt_ltap_operators.forward_project(volume_lt_ap)
    #
    # # 设置绘图
    # plt.figure(figsize=(12, 6))
    #
    # # 绘制AP投影
    # plt.subplot(1, 2, 1)  # 1行2列的第1个
    # plt.imshow(single_duo_projections_ap, cmap='gray')
    # plt.title('single to duo projection LT-AP Projection')
    #
    # # 绘制LT投影
    # plt.subplot(1, 2, 2)  # 1行2列的第2个
    # plt.imshow(single_duo_projections_lt, cmap='gray')
    # plt.title('single to duo projection LT_LT Projection')
    #
    # # 显示图像
    # plt.show()