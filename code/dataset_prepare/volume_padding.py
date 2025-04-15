

import numpy as np
import preprocessing
import os
import tifffile

def symmetric_padding_3d_volume(dicom_folder, target_shape):


    pre_volume,_,_,_=preprocessing.preprocess_dicom_images(dicom_folder)

    # 获取原始体积的形状
    original_shape = pre_volume.shape

    # 计算在每个轴上的填充量
    padding_z = (target_shape[0] - original_shape[0]) // 2

    # 创建一个新的填充后的体积
    padded_volume = np.zeros(target_shape, dtype=pre_volume.dtype)

    # 在z轴上进行对称填充
    padded_volume[padding_z:padding_z+original_shape[0], :, :] = pre_volume

    return padded_volume

def save_padded_3d_volume_as_tif(volume, output_folder, patient_name):
    # 构建保存文件路径
    output_file = os.path.join(output_folder, f"{patient_name}.tif")

    # 保存3D体积为.tif文件
    tifffile.imsave(output_file, volume)
    print(f'3D体积已保存为 {output_file}')







#
# if __name__ == '__main__':
#     dicom_folder='G:/dataset/LIDC-IDRI/LIDC-IDRI-0001'
#     target_shape=(500, 128, 128)
#     volume=symmetric_padding_3d_volume(dicom_folder, target_shape)
#     output_folder='G:/dataset_01_volume'
#     patient_name=1
#     save_padded_3d_volume_as_tif(volume, output_folder, patient_name)