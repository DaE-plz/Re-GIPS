'''

图像一致性损失（Image Consistency Loss）和几何特征一致性损失
（Geometry Feature Consistency Loss）。图像一致性损失确保从
源视图投影中提取的特征经过生成器重建后，能够恢复成相同的投影。几何特
征一致性损失确保从生成的投影中提取的几何特征与之前转换的几何特征具有
相同的表示。
'''

import torch
import torch.nn.functional as F

def image_consistency_loss(generated_image, original_image):
    """
    图像一致性损失
    :param generated_image: 由生成器重建的图像
    :param original_image: 原始图像
    :return: 损失值
    """
    return F.l1_loss(generated_image, original_image)

def geometry_feature_consistency_loss(extracted_features_tgt, transformed_features_tgt,
                                      extracted_features_src, transformed_features_src):
    """
    几何特征一致性损失
    :param extracted_features_tgt: 从目标视角合成投影中提取的几何特征（生成的图像进行encoder）
    :param transformed_features_tgt: 转换后的目标视角几何特征（原来的图像进行transform得到的）
    :param extracted_features_src: 从源视角合成投影中提取的几何特征
    :param transformed_features_src: 转换后的源视角几何特征
    :return: 损失值
    """
    loss_tgt = F.l1_loss(extracted_features_tgt, transformed_features_tgt)
    loss_src = F.l1_loss(extracted_features_src, transformed_features_src)
    return loss_tgt + loss_src

def total_consistency_loss(generated_image, original_image,
                           extracted_features_tgt, transformed_features_tgt,
                           extracted_features_src, transformed_features_src):
    """
    总一致性损失
    :param generated_image: 由生成器重建的图像
    :param original_image: 原始图像
    :param extracted_features_tgt: 从目标视角合成投影中提取的几何特征
    :param transformed_features_tgt: 转换后的目标视角几何特征
    :param extracted_features_src: 从源视角合成投影中提取的几何特征
    :param transformed_features_src: 转换后的源视角几何特征
    :return: 损失值
    """
    img_loss = image_consistency_loss(generated_image, original_image)
    feat_loss = geometry_feature_consistency_loss(extracted_features_tgt, transformed_features_tgt,
                                                  extracted_features_src, transformed_features_src)
    return img_loss + feat_loss

