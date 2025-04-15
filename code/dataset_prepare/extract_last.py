import os
import pydicom
import shutil
from collections import defaultdict
# 提取DICOM文件夹中第一个DICOM文件的信息
def extract_information_from_dicom(dicom_folder):
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()

    if not dicom_files:
        print("没有找到DICOM文件")
        return None

    try:
        ds = pydicom.dcmread(dicom_files[0])
    except Exception as e:
        print("读取DICOM文件时出错:", e)
        return None

    information = {}

    # 安全地提取信息
    def get_dicom_value(field):
        return getattr(ds, field, None)

    information["SliceThickness"] = float(get_dicom_value("SliceThickness")) if get_dicom_value("SliceThickness") else None

    return information

# 指定源目录路径和目标目录路径
source_directory_path = r'G:\dataset\LIDC-IDRI'
target_directory_path = r'G:\dataset\useless'

# 确保目标路径存在
os.makedirs(target_directory_path, exist_ok=True)

folder_dcm_counts = defaultdict(int)


# 遍历目录中的所有文件
for root, dirs, files in os.walk(source_directory_path):
    for file in files:
        if file.endswith('.dcm'):
            folder_dcm_counts[root] += 1

# 检查每个子文件夹
for folder in folder_dcm_counts:
    count = folder_dcm_counts[folder]
    info = extract_information_from_dicom(folder)

    if info and info["SliceThickness"]:
        if info["SliceThickness"] * count > 500:
            print(f"移动子文件夹 {folder} 到 {target_directory_path}")
            # 构建目标文件夹路径
            target_folder = os.path.join(target_directory_path, os.path.basename(folder))
            # 移动文件夹
            shutil.move(folder, target_folder)