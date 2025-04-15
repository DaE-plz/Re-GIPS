
import os
import pydicom
import numpy as np
from PIL import Image
import tifffile

def stack_dicom_files(dicom_folder):
    # 获取文件夹中的所有DICOM文件
    dicom_files = [os.path.join(dicom_folder, filename) for filename in os.listdir(dicom_folder) if filename.endswith('.dcm')]

    # 读取切片并按照 ImagePositionPatient 排序
    slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 检查切片尺寸
    slice_shape = list(slices[0].pixel_array.shape)
    num_slices = len(slices)

    # 初始化一个空的3D体积
    volume_shape = [num_slices] + slice_shape
    volume = np.zeros(volume_shape, dtype=np.float32)

    # 逐个读取DICOM文件、增加HU值，并填充3D体积
    for i, s in enumerate(slices):
        slice_data = s.pixel_array.astype(np.float32)  # 提取像素数据并转换为float32

        # 使用RescaleSlope和RescaleIntercept转换到HU
        # 像素值 * RescaleSlope + RescaleIntercept
        # slice_data = slice_data * s.RescaleSlope + s.RescaleIntercept
        # modified_slice_data = slice_data + 1024  # 增加HU值
        #对于slice_data，小于-1024的像素，进行加1024
        slice_data[slice_data < -1024] = -1024

        # 对于整体的slice_data,进行加1024
        slice_data += 1024
        #
        # 对于整体的slice_data,进行除以2000
        slice_data /= 60000
        # slice_data /= 2000

        volume[i, :, :] = slice_data

    return volume

def save_3d_volume_as_tif(volume, output_file):
    # 保存3D体积为.tif文件
    tifffile.imsave(output_file, volume)
    print(f'3D体积已保存为 {output_file}')





# # 实现
# dicom_folder = 'G:/dataset_01/LIDC-IDRI/LIDC-IDRI-0001'
# volume=stack_dicom_files(dicom_folder)
# output_file = 'G:/dataset_01/output_volume0001without.tif'
# save_3d_volume_as_tif(volume, output_file)
