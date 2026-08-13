# -*- coding: utf-8 -*-
"""
FPN 渐进式上采样解码器（深度/分割共享）。

从 FPN 的多尺度特征 {p2,p3,p4,p5}（stride 4/8/16/32，通道 = FPN.OUT_CHANNELS）出发
按 P5→P4→P3→P2 逐级 ×2 上采样并与对应层 skip-concat（类似 UNet 解码端）
最终上采样到目标分辨率，输出 out_channels 通道。

深度头和分割头共享这套结构，只是 out_channels 不同（深度=1，分割=num_classes）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """渐进式上采样解码器。

    Args:
        in_channels: FPN 各层通道数（默认 256）
        num_layers: 解码层数（默认 4，对应 P5→P4→P3→P2）
        out_channels: 最终输出通道（深度=1，分割=num_classes）
    """

    def __init__(self, in_channels=256, num_layers=4, out_channels=1):
        super().__init__()
        self.num_layers = num_layers

        # 每个解码块：3x3 Conv + GN + ReLU
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(num_layers)
        ])
        # skip concat 后通道翻倍，用 1x1 Conv 压回 in_channels（P5 起步无 skip）
        self.fuse = nn.ModuleList([
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
            for _ in range(num_layers - 1)
        ])
        # 最终 head：上采样到原图后，输出 out_channels 通道
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, features, output_size):
        """
        Args:
            features: dict，含 p2/p3/p4/p5
            output_size: (H, W) 目标分辨率
        Returns:
            (B, out_channels, H, W)
        """
        levels = ["p5", "p4", "p3", "p2"]
        x = features[levels[0]]  # P5 最粗层起步
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i < self.num_layers - 1:  # 非最后一块：上采样 + skip
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                skip = features[levels[i + 1]]
                if x.shape[-2:] != skip.shape[-2:]:  # 防 stride 不整除差 1 像素
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = self.fuse[i](x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.head(x)
