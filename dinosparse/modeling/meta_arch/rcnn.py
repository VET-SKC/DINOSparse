#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
整合自 https://github.com/PeizeSun/SparseR-CNN 的 detector.py
"""
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures import Boxes, Instances, ImageList

from .build import META_ARCH_REGISTRY
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..heads import DynamicHead, SetCriterion, HungarianMatcher
from ..heads.util import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

__all__ = ["SparseRCNN"]


@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """Implement SparseR-CNN (End-to-End Object Detection with Learnable Proposals)"""

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        # backbone（使用 dinosparse 的 BACKBONE_REGISTRY；输出多尺度 FPN 特征）
        self.backbone = build_backbone(cfg)

        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        # DynamicHead 的 ROIPooler 只在这些 level 上提取特征（与 ROI_HEADS.IN_FEATURES 一致）
        self.in_features = list(cfg.MODEL.ROI_HEADS.IN_FEATURES)
        # 额外 token 融合方式
        self.aux_token_fusion = cfg.MODEL.SparseRCNN.AUX_TOKEN_FUSION

        # learnable object queries：N 个可学习的 proposal boxes（cxcywh, 归一化到 0~1）
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # 检测头：DynamicHead 接收多尺度特征 + learnable proposals + proposal features
        self.head = DynamicHead(cfg, self.backbone.output_shape())

        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.aux_loss_mode = cfg.MODEL.SparseRCNN.AUX_LOSS_MODE
        self.aux_weight = cfg.MODEL.SparseRCNN.AUX_WEIGHT
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        self.giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        self.l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        self.no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT

        # matcher + criterion（匈牙利匹配 + 分类/回归 loss）
        self.matcher = HungarianMatcher(cfg,
                                        cost_class=self.class_weight,
                                        cost_bbox=self.l1_weight,
                                        cost_giou=self.giou_weight,
                                        use_focal=self.use_focal)

        # weight_dict
        # 最终层（loss_ce/loss_bbox/loss_giou）恒为基础权重，aux 层按 AUX_LOSS_MODE 缩放。
        weight_dict = {"loss_ce": self.class_weight,
                       "loss_bbox": self.l1_weight,
                       "loss_giou": self.giou_weight}
        if self.deep_supervision:
            mode, aux_w = self.aux_loss_mode, self.aux_weight
            assert mode in ("none", "linear"), f"未知 AUX_LOSS_MODE: {mode}"
            assert 0 < aux_w <= 1, f"AUX_WEIGHT 须 ∈ (0,1]，得到 {aux_w}"
            N = self.num_heads
            aux_weight_dict = {}
            for i in range(N - 1):  # i=0(最浅aux) .. N-2(最深aux) | N-1(最终层)
                if mode == "linear":
                    s = aux_w + (1 - aux_w) * i / (N - 1)
                else:  # SparseRCNN
                    s = 1.0
                aux_weight_dict.update({k + f"_{i}": v * s for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=self.no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        # pixel mean / std（preprocess_image 用）。device 迁移由 build_model / trainer 统一处理。
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)

        # 冻结开关（few-shot 微调阶段用；基类预训练时均为 False）
        self._apply_freeze(cfg)

    def _apply_freeze(self, cfg):
        """
        主干 FPN
        SparseRCNN 按ABCD四组应用冻结D组。（详见 config 注释）
        仅 BACKBONE.FREEZE 和 FREEZE_INTERACTION 两个开关。
        """
        if cfg.MODEL.BACKBONE.FREEZE:
            n_frozen = 0
            for p in self.backbone.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    n_frozen += 1
            # 显示实际 FPN 类型（DINOv3 下为 SimpleFeaturePyramid / MultilayerFPN 等）
            fpn_type = type(self.backbone).__name__
            strategy = getattr(cfg.MODEL.DINOv3, "FEATURE_STRATEGY", "n/a")
            print(f"froze backbone (DINOv3ViT + {fpn_type}, strategy={strategy}): {n_frozen} params")

        if cfg.MODEL.SparseRCNN.FREEZE_INTERACTION:
            # 冻结 D 组：head_series 每层内的 self_attn/inst_interact/linear1/linear2/norm1/norm2/norm3。
            # 匹配规则：name 含这些子串之一即冻。
            # 不命中的：cls_module / reg_module（C 组）、class_logits / bboxes_delta（B 组）、init_proposal_（A组）。
            freeze_keys = ("self_attn", "inst_interact", "linear1", "linear2", "norm1", "norm2", "norm3")
            n_frozen = 0
            for name, p in self.head.named_parameters():
                if any(k in name for k in freeze_keys):
                    if p.requires_grad:
                        p.requires_grad = False
                        n_frozen += 1
            print(f"froze SparseRCNN interaction (group D): {n_frozen} params")

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, images_whwh = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features_list = [features[f] for f in self.in_features]

        # aux_cls 和 aux_register 沿 token 维拼接
        # 只有 head 启用融合（aux_token_fusion != none）且 features dict 含有 aux_* 时才拼装
        context_tokens = None
        if self.aux_token_fusion != "none":
            aux_parts = []
            if "aux_cls" in features:
                aux_parts.append(features["aux_cls"])
            if "aux_register" in features:
                aux_parts.append(features["aux_register"])
            if aux_parts:
                context_tokens = torch.cat(aux_parts, dim=1)   # (B, num_tokens, dim)

        # proposal boxes 初始为归一化 cxcywh，乘以图像尺寸 → 像素坐标（DynamicHead 在像素坐标下工作）
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # DynamicHead：第三参数是 init_proposal_features.weight (N, d_model)，无需 unsqueeze。
        # 只传 in_features 对应的 feature（顺序/数量须与 ROIPooler 的 pooler_scales 一致）。
        outputs_class, outputs_coord = self.head(
            features_list, proposal_boxes, self.init_proposal_features.weight, context_tokens
        )
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            results = self.inference(output['pred_logits'], output['pred_boxes'], images.image_sizes)
            # do_postprocess
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        """把 detectron2 Instances list 转成 criterion/matcher 期望的 targets 格式。

        原版字段（坐标系与 loss.py/matcher 严格对应）：
            "labels": (num_gt,)         类别
            "boxes": (num_gt,4)         归一化 cxcywh（0~1）
            "boxes_xyxy": (num_gt,4)    像素坐标 xyxy（matcher/loss_giou 用）
            "image_size_xyxy": (4, )    像素 H,W,H,W（matcher 用，src 归一化）
            "image_size_xyxy_tgt":      (4, num_gt) 像素 H,W,H,W（loss_boxes 用，tgt 归一化）
            "area": (num_gt,)           面积（排序用）
        """
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = (torch.arange(self.num_classes, device=self.device).
                      unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1))

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """Normalize, pad and batch the input images.（与 GeneralizedRCNN 一致）"""
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

    def _move_to_current_device(self, x):
        # 把输入张量搬到模型的设备（pixel_mean 已由 build_model 的 .to(cfg.MODEL.DEVICE) 迁移到 GPU）。
        return x.to(self.device)
