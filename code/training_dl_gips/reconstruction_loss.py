
import torch.nn.functional as F

def reconstruction_loss(generated_tgt, ground_truth_tgt, generated_src, ground_truth_src):
    """
    重建损失
    :param generated_tgt: 图像生成器生成的目标视图投影
    :param ground_truth_tgt: 目标视图的真实投影
    :param generated_src: 图像生成器生成的源视图投影
    :param ground_truth_src: 源视图的真实投影
    :return: loss
    """
    loss_tgt = F.l1_loss(generated_tgt, ground_truth_tgt)
    loss_src = F.l1_loss(generated_src, ground_truth_src)
    return loss_tgt + loss_src

