# -*- coding: utf-8 -*-
"""
语义分割 meta-arch。

__init__ 只调 build_backbone(cfg)，拿 FPN 多尺度特征，接 FPNDecoder(out=num_classes)。
冻结逻辑同 MonoDepthMetaArch。

损失：CrossEntropy（ignore_index 排除 void）+ Dice（per-class 平均）。
评测指标（由 SemSegEvaluator 计算）：mIoU / pixel-acc / per-class IoU。

数据 GT：sem_seg（HxW long，值 0=void，1–40=类）
"""
import torch
import torch.nn.functional as F

from detectron2.structures import ImageList

from .build import META_ARCH_REGISTRY
from ..backbone import build_backbone
from ..heads import FPNDecoder


__all__ = ["SemSegMetaArch"]


@META_ARCH_REGISTRY.register()
class SemSegMetaArch(torch.nn.Module):
    """语义分割 meta-arch：FPN 特征 → FPNDecoder(out=num_classes) → 逐像素分类。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # backbone（DINOv3 + FPN），输出 {p2,p3,p4,p5}。不改动 backbone 代码。
        self.backbone = build_backbone(cfg)

        # 分割头：FPN 渐进式解码器，输出 num_classes 通道 logit
        fpn_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.num_classes = cfg.MODEL.SEM_SEG.NUM_CLASSES
        self.decoder = FPNDecoder(
            in_channels=fpn_channels,
            num_layers=cfg.MODEL.SEM_SEG.NUM_DECODER_LAYERS,
            out_channels=self.num_classes,
        )

        # loss 参数
        self.ce_weight = cfg.MODEL.SEM_SEG.CE_WEIGHT
        self.dice_weight = cfg.MODEL.SEM_SEG.DICE_WEIGHT
        self.ignore_index = cfg.MODEL.SEM_SEG.IGNORE_INDEX

        # pixel mean / std
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)

        self._apply_freeze(cfg)

    def _apply_freeze(self, cfg):
        if cfg.MODEL.BACKBONE.FREEZE:
            n_frozen = 0
            for p in self.backbone.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n_frozen += 1
            fpn_type = type(self.backbone).__name__
            strategy = getattr(cfg.MODEL.DINOv3, "FEATURE_STRATEGY", "n/a")
            print(f"[SemSeg] froze backbone (DINOv3ViT + {fpn_type}, strategy={strategy}): {n_frozen} params")

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """归一化 + pad 成 batch"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # {p2,p3,p4,p5}

        # decoder 输出到 pad 后的统一尺寸
        H, W = images.tensor.shape[-2:]
        logits = self.decoder(features, (H, W))  # (B, num_classes, H, W) pad 后统一尺寸

        if self.training:
            # batch 内各图 resize 后尺寸不同，backbone 已 pad 统一；GT 逐张 resize 对齐。
            gt_list = []
            for x in batched_inputs:
                gt = x["sem_seg"].long().to(self.device)  # (h,w)
                if gt.shape[-2:] != (H, W):
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                        mode="nearest"
                    ).squeeze(0).squeeze(0).long()
                gt_list.append(gt)
            gt_seg = torch.stack(gt_list, dim=0)  # (B,H,W)

            # NYU labels40 编码: 0=void, 1..40=类。
            # head 输出 num_classes=40 通道(索引 0..39)。
            # 把 1..40 映射到 0..39：gt_seg - 1。
            # void(0) -> -1，用 ignore_index=-1 排除。
            gt_seg = gt_seg - 1

            loss_ce = F.cross_entropy(logits, gt_seg, ignore_index=self.ignore_index)
            loss_dice = dice_loss(logits, gt_seg, self.num_classes, self.ignore_index)

            loss_dict = {
                "loss_ce": loss_ce * self.ce_weight,
                "loss_dice": loss_dice * self.dice_weight,
            }
            return loss_dict
        else:
            # 推理：裁掉 padding，还原到原图尺寸，输出 argmax 类别
            pred = logits.argmax(dim=1)  # (B, H, W)
            results = []
            for i, in_size in enumerate(images.image_sizes):
                h, w = in_size
                pred_i = pred[i, :h, :w]
                orig_h = batched_inputs[i].get("height", h)
                orig_w = batched_inputs[i].get("width", w)
                if (h, w) != (orig_h, orig_w):
                    pred_i = F.interpolate(
                        pred_i.unsqueeze(0).unsqueeze(0).float(), size=(orig_h, orig_w),
                        mode="nearest"
                    ).squeeze(0).squeeze(0).long()
                results.append({"sem_seg": pred_i.detach().cpu()})
            return results


def dice_loss(logits, target, num_classes, ignore_index, eps=1e-6):
    """Soft Dice loss（per-class 平均）。

    对每个类在 valid 像素上算 Dice = 2|P∩T| / (|P|+|T|)，loss = 1 - mean(Dice)。
    ignore_index 像素在所有计算中排除。
    target 已经过 -1 映射：有效类为 0..num_classes-1，void=-1(=ignore_index)。
    """
    # 概率化
    probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

    # 构造 valid mask，排除 ignore_index（void=-1）
    valid = (target != ignore_index)  # (B, H, W)

    # valid 处的类别范围 clamp 到 [0, num_classes-1] 以便 one_hot（void 已被 valid 排除）
    target_clamped = target.clamp(min=0, max=num_classes - 1)
    target_onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()  # (B,C,H,W)

    # 应用 valid mask
    valid_f = valid.unsqueeze(1).float()  # (B,1,H,W)
    probs = probs * valid_f
    target_onehot = target_onehot * valid_f

    dims = (0, 2, 3)  # 在 batch + 空间维上求和
    intersection = (probs * target_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()
