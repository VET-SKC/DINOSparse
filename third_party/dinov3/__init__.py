"""DINOv3 即插即用封装（基于 HuggingFace transformers）。

主要接口：
    from third_party.dinov3 import DINOv3ViT
    vit = DINOv3ViT(model_path="data/pretrain_weights/dinov3-vitb16-pretrain-lvd1689m",
                    out_indices=[12], freeze=True)
    out = vit(images)   # images: 已归一化 RGB (B,3,H,W), H/W 为 patch_size 整数倍

详见 wrapper.DINOv3ViT 的文档字符串。
"""
from .wrapper import DINOv3ViT

__all__ = ["DINOv3ViT"]
