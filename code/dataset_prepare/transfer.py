import os
import shutil
#  将.pt & .tif 分别转移

# 设置源目录和目标目录
source_dir = 'G:/dataset/projection_lt'
target_dir = 'G:/dataset/projection_lt_tif'

# 检查目标目录是否存在，如果不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源目录中的所有文件
for file in os.listdir(source_dir):
    # 检查文件扩展名是否为.tif
    if file.endswith('.tif'):
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)

        # 移动文件
        shutil.move(source_file, target_file)

print('所有.tiff文件已经成功移动到目标目录。')