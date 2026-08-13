# -*- coding: utf-8 -*-
"""
单目深度估计 evaluator。

复用本地 DatasetEvaluator 接口（reset/process/evaluate）。

指标定义（NYU Depth V2 标准评价体系）:
  - AbsRel  = mean(|pred - gt| / gt)              绝对相对误差，无量纲，最常用
  - SqRel   = mean((pred - gt)^2 / gt)            平方相对误差
  - RMSE    = sqrt(mean((pred - gt)^2))           均方根误差，单位米
  - RMSElog = sqrt(mean((log(pred) - log(gt))^2)) 对数均方根误差
  - log10   = mean(|log10(pred) - log10(gt)|)
  - δ < t^k:  max(pred/gt, gt/pred) < t^k 的像素占比，k=1,2,3（阈值准确率）
NYU 上典型基线：AbsRel≈0.1，δ<1.25≈0.85

GT 来源：dataset_dict["depth"]（DepthDatasetMapper 产出，float32 米）
pred 来源：模型推理输出 [{"depth": HxW}]
两者尺寸可能因 ResizeShortestEdge 不同，用 nearest 对齐到 GT 尺寸。

后处理（process 内，模型与训练均不受影响）:
  ① 逐图 median 尺度对齐：scale = median(gt)/median(pred)，pred *= scale。
     单目深度（尤其尺度无关 loss 训练的）只能学到相对结构，全局缩放 c 不定；
     评测时每张图用 GT 中位数把 c 补回来，衡量相对精度。
     SILog / gradient_matching 对全局缩放 c 不敏感，训练会收敛到 pred≈c·gt，
     不做这一步 AbsRel 会被 |c−1| 主导（如 c≈0.16或1.84 → AbsRel≈0.84）。
  ② 中心裁剪（Eigen/Garg）：剔除 NYU 深度图不可靠的外圈像素（Kinect 边缘畸变/投影漏洞）。
两者都可由 cfg.MODEL.MONO_DEPTH.EVAL_MEDIAN_ALIGN / EVAL_CROP 开关控制。
"""
import logging
from collections import OrderedDict

import numpy as np

from .evaluator import DatasetEvaluator


# NYU 480×640 上的评测裁剪框 (top, bottom, left, right)，行优先。
# 源自 Eigen / Garg ？
# 以 480×640 为参考，_crop_mask 按实际尺寸等比缩放。
_NYU_CROP_BOUNDS = (45, 471, 41, 601)


class MonoDepthEvaluator(DatasetEvaluator):
    """单目深度估计 evaluator。"""

    def __init__(self, dataset_name, output_folder=None,
                 min_depth=1e-3, max_depth=10.0, median_align=False, crop=False):
        """
        Args:
            min_depth / max_depth: valid 像素深度区间（米，闭区间，与训练 loss 一致）。
            median_align: 是否逐图 median(gt)/median(pred) 尺度对齐（在 eval 时消除尺度漂移）。
            crop: 是否做 NYU 中心裁剪。
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_folder = output_folder
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._median_align = median_align
        self._crop = crop
        self.reset()

    def reset(self):
        # 累积各指标的和与计数（避免一次性存所有像素，1449 张内存可控但用累积更通用）
        self._num_valid = 0
        self._abs_rel = 0.0
        self._sq_rel = 0.0
        self._rmse = 0.0
        self._rmse_log = 0.0
        self._log10 = 0.0
        self._delta1 = 0.0  # δ < 1.25
        self._delta2 = 0.0  # δ < 1.25^2
        self._delta3 = 0.0  # δ < 1.25^3
        # 逐图 median(pred)/median(gt) 的累积（对齐前的原始比值，用于验证尺度漂移程度）
        # 若模型定尺度好 → 趋近 1；若 pred≈c·gt → 趋近 c
        self._scale_ratio_sum = 0.0
        self._scale_ratio_count = 0

    def _crop_mask(self, shape):
        """按 _NYU_CROP_BOUNDS 生成中心裁剪 bool mask（以 480×640 为参考等比缩放到 shape）。"""
        H, W = shape
        top, bottom, left, right = _NYU_CROP_BOUNDS
        sh, sw = H / 480.0, W / 640.0
        t = int(round(top * sh))
        b = int(round(bottom * sh))
        l = int(round(left * sw))
        r = int(round(right * sw))
        mask = np.zeros((H, W), dtype=bool)
        mask[t:b, l:r] = True
        return mask

    def process(self, inputs, outputs):
        for input_dict, output_dict in zip(inputs, outputs):
            gt = input_dict["depth"]                       # (H,W) float32 米
            pred = output_dict["depth"]                    # (H,W) float32 米
            # 统一到 numpy
            gt = gt.cpu().numpy().astype(np.float64)
            pred = pred.cpu().numpy().astype(np.float64)

            # 尺寸对齐（pred 可能是 backbone 分辨率，GT 是 mapper 分辨率）
            if pred.shape != gt.shape:
                from PIL import Image
                pred = np.asarray(
                    Image.fromarray(pred.astype(np.float32)).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    )
                ).astype(np.float64)

            # 1. valid：先只按 GT 过滤
            # 用闭区间，与训练 loss(depth.py: valid = gt>=min & gt<=max)一致
            # valid = (gt >= self._min_depth) & (gt <= self._max_depth) \
            #         & (pred >= self._min_depth) & (pred <= self._max_depth)
            valid = (gt >= self._min_depth) & (gt <= self._max_depth)
            if not valid.any():
                continue

            # 2. 逐图 median 尺度对齐
            if self._median_align:
                med_gt = np.median(gt[valid])
                med_pred = np.median(pred[valid])
                if med_pred > 0:
                    # 记录对齐前的原始比值 c = median(pred)/median(gt)，供 evaluate 日志验证
                    self._scale_ratio_sum += float(med_pred / med_gt)
                    self._scale_ratio_count += 1
                    pred = pred * (med_gt / med_pred)

            # 3. clip 到合法区间（median 对齐后个别像素可能越界）
            pred = np.clip(pred, self._min_depth, self._max_depth)

            # 4. 中心裁剪：剔除 NYU 深度图的外圈不可靠部分
            if self._crop:
                valid = valid & self._crop_mask(gt.shape)

            gt_v = gt[valid]
            pred_v = pred[valid]
            n = gt_v.size
            if n == 0:
                continue

            # 5. 各指标累加（用 sum 存，evaluate 时除以总 valid 数）
            self._num_valid += n
            self._abs_rel += np.sum(np.abs(pred_v - gt_v) / gt_v)
            self._sq_rel += np.sum((pred_v - gt_v) ** 2 / gt_v)
            self._rmse += np.sum((pred_v - gt_v) ** 2)

            log_gt = np.log(gt_v)
            log_pred = np.log(np.clip(pred_v, 1e-8, None))
            self._rmse_log += np.sum((log_pred - log_gt) ** 2)
            self._log10 += np.sum(np.abs(np.log10(pred_v) - np.log10(gt_v)))

            # δ 阈值
            thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
            self._delta1 += np.sum(thresh < 1.25)
            self._delta2 += np.sum(thresh < 1.25 ** 2)
            self._delta3 += np.sum(thresh < 1.25 ** 3)

    def evaluate(self):
        n = self._num_valid
        if n == 0:
            self._logger.warning("[MonoDepthEvaluator] 无有效像素，跳过评测")
            return None

        ret = OrderedDict()
        stats = OrderedDict({
            "AbsRel": self._abs_rel / n,
            "SqRel": self._sq_rel / n,
            "RMSE": np.sqrt(self._rmse / n),
            "RMSElog": np.sqrt(self._rmse_log / n),
            "log10": self._log10 / n,
            "delta1 (δ<1.25)": self._delta1 / n,
            "delta2 (δ<1.25^2)": self._delta2 / n,
            "delta3 (δ<1.25^3)": self._delta3 / n,
        })

        # 逐图原始尺度比值（median(pred)/median(gt) 均值）——验证尺度漂移是否被定住
        if self._scale_ratio_count > 0:
            stats["median_ratio (pred/gt)"] = self._scale_ratio_sum / self._scale_ratio_count
        stats["num_valid_pixels"] = int(n)

        ret["depth"] = stats
        self._logger.info(
            f"[MonoDepthEvaluator] {self._dataset_name} 深度评测结果 "
            f"(median_align={self._median_align}, crop={self._crop}):"
        )
        for k, v in ret["depth"].items():
            self._logger.info(f"  {k}: {v}")
        return ret
