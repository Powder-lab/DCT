import torch
import torch.nn.functional as F
from torchvision.models import vgg19
from transformers import CLIPModel
import math

# 初始化特征提取器
def initialize_feature_extractors(clip_model_path="./clip-vit-base-patch32", vgg_pretrained=True, target_resolution=512):
    """
    初始化CLIP和VGG模型，用于特征提取
    Args:
        clip_model_path (str): CLIP模型路径
        vgg_pretrained (bool): 是否加载VGG的预训练权重
        target_resolution (int): 目标图像分辨率，用于动态调整CLIP模型的嵌入
    Returns:
        tuple: (CLIP模型, VGG模型)
    """
    print(f"Loading CLIP model from: {clip_model_path}")
    clip_model = CLIPModel.from_pretrained(clip_model_path).cuda().half()
    clip_model = modify_clip_position_embeddings(clip_model, target_resolution)
    clip_model.eval()

    print(f"Loading VGG model. Pretrained: {vgg_pretrained}")
    vgg = vgg19(pretrained=vgg_pretrained).features[:16].cuda().half()  # 提取VGG的前16层特征
    vgg.eval()

    return clip_model, vgg

def modify_clip_position_embeddings(clip_model, target_resolution):
    """
    修改 CLIP 模型的位置嵌入以适配目标分辨率。
    Args:
        clip_model (CLIPModel): 原始 CLIP 模型。
        target_resolution (int): 目标分辨率（如 512）。
    """
    patch_size = clip_model.config.vision_config.patch_size  # 获取 Vision Transformer 的 patch 大小
    num_patches_per_dim = target_resolution // patch_size
    num_patches = num_patches_per_dim ** 2

    # 获取原始位置嵌入
    original_position_embeddings = clip_model.vision_model.embeddings.position_embedding.weight
    original_num_patches = original_position_embeddings.shape[0]
    hidden_size = clip_model.config.vision_config.hidden_size  # 从 vision_config 中获取 hidden_size

    # 动态计算最接近的完全平方数维度
    original_dim = int(math.sqrt(original_num_patches))
    if original_dim ** 2 != original_num_patches:
        print(f"Warning: Original number of patches ({original_num_patches}) is not a perfect square. "
              f"Adjusting to {original_dim ** 2}.")
        original_dim = math.floor(math.sqrt(original_num_patches))

    if num_patches != original_num_patches:
        print(f"Resizing position embeddings from {original_num_patches} to {num_patches}")

        # 重新插值位置嵌入
        new_position_embeddings = F.interpolate(
            original_position_embeddings[: original_dim ** 2].view(
                1, original_dim, original_dim, hidden_size
            ).permute(0, 3, 1, 2),  # 形状为 (1, hidden_size, height, width)
            size=(num_patches_per_dim, num_patches_per_dim),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, hidden_size)

        # 增加 class_token 对应的位置嵌入
        class_token_embedding = original_position_embeddings[-1:]  # 使用原始 class_token 的嵌入
        new_position_embeddings = torch.cat([class_token_embedding, new_position_embeddings], dim=0)

        # 替换位置嵌入
        clip_model.vision_model.embeddings.position_embedding = torch.nn.Embedding.from_pretrained(
            new_position_embeddings, freeze=False
        )

        # 重新生成 position_ids
        clip_model.vision_model.embeddings.position_ids = torch.arange(
            num_patches + 1, dtype=torch.long, device=new_position_embeddings.device
        ).unsqueeze(0)

    return clip_model

def extract_features(images, model_type, clip_model=None, vgg=None):
    """
    提取语义或纹理特征，支持批量化处理视频帧
    """
    is_video = len(images.shape) == 5  # 检查是否是视频输入

    if is_video:
        # 将时间维度与批次维度合并
        batch_size, channels, num_frames, height, width = images.shape
        images = images.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)  # (B*T, C, H, W)

    with torch.no_grad():
        if model_type == "clip":
            features = clip_model.get_image_features(images)  # CLIP特征
            features = F.normalize(features, dim=-1)
        elif model_type == "vgg":
            features = vgg(images)  # VGG特征
        else:
            raise ValueError("Unsupported model_type. Use 'clip' or 'vgg'.")

    if is_video:
        # 恢复时间维度并确保特征对齐
        features = features.view(batch_size, num_frames, -1)  # (B, T, feature_dim)
        features = features.mean(dim=1)  # 对时间维度求平均，使特征一致

    return features


def feature_consistency_loss(gen_features, target_features, weight=1.0):
    """
    计算生成图像与目标图像之间的特征一致性损失
    """
    print(f"Generated features shape: {gen_features.shape}")
    print(f"Target features shape: {target_features.shape}")

    # 如果生成特征只有一个样本，则重复它以匹配目标特征的维度
    if gen_features.shape[0] == 1 and target_features.shape[0] != 1:
        gen_features = gen_features.repeat(target_features.shape[0], 1)

    # 确保生成特征和目标特征的形状一致
    if gen_features.shape != target_features.shape:
        # 扁平化目标特征以匹配生成特征的形状
        target_features = target_features.reshape(target_features.size(0), -1)  # [24, 4194304]
        #raise ValueError("Generated features and target features must have the same shape.")

    return weight * F.mse_loss(gen_features, target_features)


# 动态调整权重
def adjust_weights(timestep, total_timesteps, mode="exponential", semantic_grad=0.7, texture_grad=0.3, mu=None, sigma=1.0):
    """
    根据时间步调整语义和纹理一致性权重，支持多种调整策略，包括线性、余弦、指数和高斯分布结合梯度动态调整。
    Args:
        timestep (int): 当前时间步
        total_timesteps (int): 总时间步数
        mode (str): 权重调整模式，可选 "linear"、"cosine"、"exponential"、"gaussian_gradient"
        semantic_grad (float): 语义损失梯度范数，仅在 mode="gaussian_gradient" 时使用
        texture_grad (float): 纹理损失梯度范数，仅在 mode="gaussian_gradient" 时使用
        mu (float): 高斯分布的中心，默认在时间中点，仅在 mode="gaussian_gradient" 时使用
        sigma (float): 高斯分布的宽度，仅在 mode="gaussian_gradient" 时使用

    Returns:
        tuple: (semantic_weight, texture_weight)
    """
    if timestep > total_timesteps or timestep < 0:
        raise ValueError("Timestep must be in the range [0, total_timesteps].")

    # 确保 timestep 和 total_timesteps 是 Tensor
    timestep = torch.tensor(timestep, dtype=torch.float32)
    total_timesteps = torch.tensor(total_timesteps, dtype=torch.float32)

    if mode == "linear":
        semantic_weight = 1.0 - timestep / total_timesteps
        texture_weight = timestep / total_timesteps
    elif mode == "cosine":
        semantic_weight = 0.5 * (1 + torch.cos(torch.pi * timestep / total_timesteps))
        texture_weight = 1.0 - semantic_weight
    elif mode == "exponential":
        semantic_weight = torch.exp(-timestep / total_timesteps)
        texture_weight = 1.0 - semantic_weight
    elif mode == "gaussian_gradient":
        if semantic_grad is None or texture_grad is None:
            raise ValueError("For 'gaussian_gradient' mode, semantic_grad and texture_grad must be provided.")
        if mu is None:
            mu = total_timesteps / 2  # 默认将高斯分布中心设置为时间步中点

        # 归一化梯度
        epsilon = 1e-8
        grad_sum = max(semantic_grad + texture_grad, epsilon)
        semantic_ratio = semantic_grad / grad_sum
        texture_ratio = texture_grad / grad_sum

        # 限制指数运算的范围，避免溢出
        max_exp = 20
        min_exp = -20
        gaussian_semantic = torch.exp(
            torch.clamp(-((timestep - mu) ** 2) / (2 * sigma ** 2) * semantic_ratio, min=min_exp, max=max_exp))
        gaussian_texture = torch.exp(
            torch.clamp(-((timestep - mu) ** 2) / (2 * sigma ** 2) * texture_ratio, min=min_exp, max=max_exp))

        # 计算权重
        semantic_weight = gaussian_semantic / (gaussian_semantic + gaussian_texture)
        texture_weight = 1.0 - semantic_weight
    else:
        raise ValueError("Unsupported mode. Use 'linear', 'cosine', 'exponential', or 'gaussian_gradient'.")

    print(f"Timestep {timestep}/{total_timesteps}: Semantic weight {semantic_weight}, Texture weight {texture_weight}")
    return semantic_weight, texture_weight


# 测试 adjust_weights 函数
semantic_weight, texture_weight = adjust_weights(
    timestep=500,
    total_timesteps=1000,
    mode="gaussian_gradient",
    semantic_grad=0.7,
    texture_grad=0.3,
    sigma=0.5
)

print(f"Final Weights - Semantic: {semantic_weight}, Texture: {texture_weight}")
