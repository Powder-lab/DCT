import torch
import numpy as np
import cv2
import torch.nn.functional as F


# 计算傅里叶变换并返回幅度谱和相位谱
def compute_fft(image):
    """
    计算图像的傅里叶变换，返回幅度谱和相位谱。
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将低频成分移动到频谱中心
    magnitude = np.abs(fshift)  # 幅度谱
    phase = np.angle(fshift)  # 相位谱
    return magnitude, phase


# 对图像进行多尺度傅里叶变换，返回每个尺度下的幅度谱和相位谱
def multi_scale_fft(image, scales=[1, 2, 4]):
    """
    对图像进行多尺度傅里叶变换，返回每个尺度下的幅度谱和相位谱。
    """
    scale_magnitudes = []
    scale_phases = []

    for scale in scales:
        resized_image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
        magnitude, phase = compute_fft(resized_image)
        scale_magnitudes.append(magnitude)
        scale_phases.append(phase)

    return scale_magnitudes, scale_phases


# 基于频域特征调整语义和纹理损失的权重
def frequency_based_weight_adjustment(semantic_loss, texture_loss, image_semantic, image_texture):
    """
    基于频域特征调整语义和纹理损失的权重。
    """
    # 计算图像的频域特征
    mag_semantic, phase_semantic = compute_fft(image_semantic)
    mag_texture, phase_texture = compute_fft(image_texture)

    # 计算频域能量的总和
    S_semantic = np.sum(mag_semantic)
    S_texture = np.sum(mag_texture)

    # 多尺度分析
    scale_mags_semantic, scale_phases_semantic = multi_scale_fft(image_semantic)
    scale_mags_texture, scale_phases_texture = multi_scale_fft(image_texture)

    # 计算每个尺度下的能量总和
    S_semantic_scaled = np.sum([np.sum(mag) for mag in scale_mags_semantic])
    S_texture_scaled = np.sum([np.sum(mag) for mag in scale_mags_texture])

    # 综合频域信息来动态调整权重
    alpha = (S_semantic + S_semantic_scaled) / (S_semantic + S_texture + S_semantic_scaled + S_texture_scaled + 1e-8)

    # 最终损失
    total_loss = alpha * semantic_loss + (1 - alpha) * texture_loss

    return total_loss


# 频域调整损失函数
def adjusted_loss_function(semantic_loss, texture_loss, image_semantic, image_texture):
    """
    计算调整后的总损失，通过频域分析结合语义损失和纹理损失。
    """
    # 计算基于频域的权重
    adjusted_loss = frequency_based_weight_adjustment(semantic_loss, texture_loss, image_semantic, image_texture)
    return adjusted_loss


# 示例图像
image_semantic = np.random.rand(256, 256)  # 假设为语义图像
image_texture = np.random.rand(256, 256)  # 假设为纹理图像

# 假设的语义损失和纹理损失
semantic_loss = 1.0
texture_loss = 0.5

# 计算频域调整后的总损失
adjusted_loss = adjusted_loss_function(semantic_loss, texture_loss, image_semantic, image_texture)
print(f"Adjusted Total Loss: {adjusted_loss}")
