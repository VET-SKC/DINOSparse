# -*- coding: utf-8 -*-
"""
单目深度估计 meta-arch。

__init__ 只调 build_backbone(cfg)，拿到 FPN 多尺度特征 {p2,p3,p4,p5}，然后接一个渐进式上采样深度头。
冻结模式复用全局 MODEL.BACKBONE.FREEZE：
  True  -> 冻 ViT+FPN，只训深度头（"只训 head"）
  False -> 解冻 FPN（ViT 仍由 MODEL.DINOv3.FREEZE 冻结）

损失：SILog (scale-invariant log loss) + 梯度匹配 loss + 尺度感知 loss（限定绝对尺度）
SILog 与梯度匹配均尺度无关（对全局缩放 c 不敏感），虽在评测端做逐图 median 尺度对齐，
但若要让模型直接输出绝对米制深度，需开启 MODEL.MONO_DEPTH.USE_SCALE_LOSS。
评测指标（由 MonoDepthEvaluator 计算）：AbsRel / RMSE / δ<1.25 等。
"""
import torch
import torch.nn.functional as F

from detectron2.structures import ImageList

from .build import META_ARCH_REGISTRY
from ..backbone import build_backbone
from ..heads import FPNDecoder


__all__ = ["MonoDepthMetaArch"]


@META_ARCH_REGISTRY.register()
class MonoDepthMetaArch(torch.nn.Module):
    """单目深度估计 meta-arch：FPN 特征 → FPNDecoder(out=1) → 逐像素深度。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # backbone（DINOv3 + FPN），输出 {p2,p3,p4,p5}。不改动 backbone 代码。
        self.backbone = build_backbone(cfg)

        # 深度头：FPN 渐进式解码器，输出 1 通道
        fpn_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.decoder = FPNDecoder(
            in_channels=fpn_channels,
            num_layers=cfg.MODEL.MONO_DEPTH.NUM_DECODER_LAYERS,
            out_channels=1,
        )

        # loss 相关参数
        self.use_grad_match = cfg.MODEL.MONO_DEPTH.USE_GRAD_MATCH
        self.silog_weight = cfg.MODEL.MONO_DEPTH.SILOG_WEIGHT
        self.grad_match_weight = cfg.MODEL.MONO_DEPTH.GRAD_MATCH_WEIGHT
        self.min_depth = cfg.MODEL.MONO_DEPTH.MIN_DEPTH
        self.max_depth = cfg.MODEL.MONO_DEPTH.MAX_DEPTH
        self.use_scale_loss = cfg.MODEL.MONO_DEPTH.USE_SCALE_LOSS
        self.scale_loss_weight = cfg.MODEL.MONO_DEPTH.SCALE_LOSS_WEIGHT

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
            print(f"[MonoDepth] froze backbone (DINOv3ViT + {fpn_type}, strategy={strategy}): {n_frozen} params")

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

        # decoder 输出到 pad 后的统一尺寸（backbone 对 batch 内所有图 pad 成同尺寸）
        H, W = images.tensor.shape[-2:]
        depth_logit = self.decoder(features, (H, W))  # (B,1,H,W) pad 后统一尺寸
        depth_pred = self.max_depth * torch.sigmoid(depth_logit)  # 限制到 (0, max_depth)

        if self.training:
            # batch 内各图 resize 后尺寸不同（ResizeShortestEdge 多尺度），
            # backbone 已 pad 成统一尺寸；GT 逐张 resize 到 pred 的统一尺寸后堆叠。
            gt_list = []
            for x in batched_inputs:
                gt = x["depth"].to(self.device)  # (h,w)
                if gt.shape[-2:] != (H, W):
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0), size=(H, W),
                        mode="nearest"
                    ).squeeze(0).squeeze(0)
                gt_list.append(gt)
            gt_depths = torch.stack(gt_list, dim=0)  # (B,H,W)

            # valid mask: GT 在 [min_depth, max_depth] 内
            valid = (gt_depths >= self.min_depth) & (gt_depths <= self.max_depth)

            loss_silog = silog_loss(depth_pred.squeeze(1), gt_depths, valid)
            loss_dict = {"loss_silog": loss_silog * self.silog_weight}

            if self.use_scale_loss:
                loss_scale = scale_loss(depth_pred.squeeze(1), gt_depths, valid)
                loss_dict["loss_scale"] = loss_scale * self.scale_loss_weight

            if self.use_grad_match:
                loss_grad = gradient_matching_loss(depth_pred.squeeze(1), gt_depths, valid)
                loss_dict["loss_grad"] = loss_grad * self.grad_match_weight

            return loss_dict
        else:
            # 推理：按每张图 pad 前的真实尺寸裁剪 pred，输出原图（resize 后）尺寸
            results = []
            for i, in_size in enumerate(images.image_sizes):
                # 裁掉 padding 区域
                h, w = in_size
                pred_i = depth_pred[i, 0, :h, :w]
                # 还原到 dataset_dict 记录的原始 height/width（mapper resize 前的尺寸）
                orig_h = batched_inputs[i].get("height", h)
                orig_w = batched_inputs[i].get("width", w)
                if (h, w) != (orig_h, orig_w):
                    pred_i = F.interpolate(
                        pred_i.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
                        mode="bilinear", align_corners=False
                    ).squeeze(0).squeeze(0)
                results.append({"depth": pred_i.detach().cpu()})
            return results


def silog_loss(pred, gt, valid):
    """Scale-Invariant Log loss (Eigen et al. 2014)。

    L = sqrt( (1/n) Σ d_i²  -  (1/n²) (Σ d_i)² ),  d_i = log(pred) - log(gt)
    只在 valid 像素上计算。
    """
    eps = 1e-8
    log_pred = torch.log(pred.clamp_min(eps))
    log_gt = torch.log(gt.clamp_min(eps))
    d = log_pred - log_gt
    d = d[valid]
    n = d.numel()
    if n == 0:
        return pred.sum() * 0.0  # 防空 batch
    term1 = (d ** 2).mean()
    term2 = (d.sum() ** 2) / (n ** 2)
    return torch.sqrt(term1 - term2 + eps)


def gradient_matching_loss(pred, gt, valid):
    """梯度匹配 loss：对 log-depth 的 x/y 方向梯度做 L1。

    鼓励预测深度的边缘与 GT 对齐，常作为 SILog 的补充。
    """
    eps = 1e-8
    log_pred = torch.log(pred.clamp_min(eps))
    log_gt = torch.log(gt.clamp_min(eps))
    v = valid.float()
    # x 方向梯度
    pred_dx = log_pred[..., :, 1:] - log_pred[..., :, :-1]
    gt_dx = log_gt[..., :, 1:] - log_gt[..., :, :-1]
    v_dx = v[..., :, 1:] * v[..., :, :-1]
    # y 方向梯度
    pred_dy = log_pred[..., 1:, :] - log_pred[..., :-1, :]
    gt_dy = log_gt[..., 1:, :] - log_gt[..., :-1, :]
    v_dy = v[..., 1:, :] * v[..., :-1, :]
    loss_x = (torch.abs(pred_dx - gt_dx) * v_dx).sum() / (v_dx.sum() + eps)
    loss_y = (torch.abs(pred_dy - gt_dy) * v_dy).sum() / (v_dy.sum() + eps)
    return (loss_x + loss_y) / 2


def scale_loss(pred, gt, valid):
    """尺度感知 loss：valid 像素上 (log 深度均值差) 的绝对值。

    L = | mean_valid( log(pred) − log(gt) ) |

    动机：SILog = sqrt(mean(d²) − mean(d)²) 在公式里先减去 mean(d)²，对全局缩放 c 完全不敏感（pred=c·gt 时 SILog≡0）；
    gradient_matching 作用在 log 梯度上，常数 log(c) 在相邻像素差分里也被消掉，同样尺度无关。
    两个 loss 都无法定尺度 → 模型收敛到 pred≈c·gt，训练 loss 健康但绝对指标灾难（尺度漂移）。

    本项直接惩罚 |mean(d)|，正好补上 SILog 丢弃的那一维尺度信息：
      pred=c·gt 时 d=log(c)（逐像素常数），mean(d)=log(c)，L=|log(c)|，最小点在 c=1。
    与 SILog / gradient_matching（皆尺度无关）正交。
    """
    eps = 1e-8
    log_pred = torch.log(pred.clamp_min(eps))
    log_gt = torch.log(gt.clamp_min(eps))
    d = log_pred - log_gt
    d = d[valid]
    n = d.numel()
    if n == 0:
        return pred.sum() * 0.0  # 防空 batch
    return d.mean().abs()
