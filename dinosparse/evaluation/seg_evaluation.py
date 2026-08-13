# -*- coding: utf-8 -*-
"""
语义分割 evaluator。

复用本地 DatasetEvaluator 接口（reset/process/evaluate）。

指标定义（语义分割标准评价体系）:
  用混淆矩阵 M（C×C，M[i][j] = GT 为 i 但预测为 j 的像素数）计算：
  - pixel_acc (PA)  = Σ M[i][i] / Σ M[i][*]                 总体像素准确率
  - mean_acc (MPA)  = mean_i( M[i][i] / Σ M[i][*] )         每类准确率的平均
  - IoU_i           = M[i][i] / (Σ M[i][*] + Σ M[*][i] - M[i][i])   类 i 的交并比
  - mIoU            = mean_i(IoU_i)                         各类 IoU 的平均（最常用）

NYU labels40 编码：0=void，1-40=类。评测时 void(0) 排除。
模型推理输出已在 argmax 后映射回 0-39（SegMetaArch 内部 GT-1），这里评测时需对齐：
  统一约定——evaluator 内部把 GT 的 1-40 和 pred 的 0-39 都对到同一空间。
  为避免混淆，这里采用：pred 和 GT 都用"原始标签空间"（0=void，1-40=类）。
  SemSegMetaArch 推理输出是 argmax(0..39)，这里 +1 映射回 1-40；GT 本就是 1-40。
  void(0) 在 GT 和 pred 中都排除。
"""
import logging
from collections import OrderedDict

import numpy as np

from .evaluator import DatasetEvaluator


class SemSegEvaluator(DatasetEvaluator):
    """语义分割 evaluator，输出 mIoU / pixel-acc / per-class IoU。"""

    def __init__(self, dataset_name, output_folder=None, num_classes=40):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_folder = output_folder
        # 标签空间：0=void(ignore)，1..num_classes=有效类
        self._num_classes = num_classes
        # 混淆矩阵尺寸 = num_classes+1（多一个 void 槽位，index 0 = void）
        self._matrix_size = num_classes + 1
        self.reset()

    def reset(self):
        # 混淆矩阵 M[i][j]：GT 为 i、pred 为 j 的像素数（含 void 行/列）
        self._confusion = np.zeros((self._matrix_size, self._matrix_size), dtype=np.int64)

    def _fast_hist(self, gt, pred):
        """累积单张图的混淆矩阵。gt/pred 用原始标签空间（0=void，1..C=类）。"""
        # 只统计 valid 像素：gt >= 0（这里 GT 不含负，全保留；void=0 也会进矩阵但评测时排除）
        mask = (gt >= 0) & (gt < self._matrix_size) & (pred >= 0) & (pred < self._matrix_size)
        gt = gt[mask].astype(np.int64)
        pred = pred[mask].astype(np.int64)
        # 二维 bincount -> 混淆矩阵
        idx = gt * self._matrix_size + pred
        hist = np.bincount(idx, minlength=self._matrix_size ** 2)
        return hist.reshape(self._matrix_size, self._matrix_size)

    def process(self, inputs, outputs):
        for input_dict, output_dict in zip(inputs, outputs):
            gt = input_dict["sem_seg"].cpu().numpy()        # (H,W) 0=void，1-40=类
            pred = output_dict["sem_seg"].cpu().numpy()     # (H,W) 0-39 (argmax)
            # pred 从 argmax 空间(0-39) 映射回原始标签空间(1-40)
            pred = pred + 1

            # 尺寸对齐（pred 可能与 GT 尺寸不同）
            if pred.shape != gt.shape:
                from PIL import Image
                # nearest 插值，保持类别标签整数性
                pred = np.asarray(
                    Image.fromarray(pred.astype(np.uint8)).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    )
                )

            self._confusion += self._fast_hist(gt, pred)

    def evaluate(self):
        M = self._confusion
        # 有效类 = 1..num_classes（排除 index 0 = void）
        valid_idx = np.arange(1, self._matrix_size)  # [1..40]

        # pixel accuracy（排除 void）
        tp_total = M[valid_idx][:, valid_idx].diagonal().sum()
        total = M[valid_idx][:, :].sum()
        pixel_acc = float(tp_total) / float(total) if total > 0 else 0.0

        # per-class IoU + mean_acc
        iou_per_class = []
        acc_per_class = []
        for c in valid_idx:
            tp = M[c, c]
            gt_count = M[c, :].sum()      # GT 为 c 的总数（已经不含 void）
            pred_count = M[1:, c].sum()   # GT 非 void 且预测为 c 的像素数
            union = gt_count + pred_count - tp
            iou = float(tp) / float(union) if union > 0 else float("nan")
            acc = float(tp) / float(gt_count) if gt_count > 0 else float("nan")
            iou_per_class.append(iou)
            acc_per_class.append(acc)

        # mIoU：忽略该数据集未出现的类（nan）
        valid_iou = [x for x in iou_per_class if not np.isnan(x)]
        miou = float(np.mean(valid_iou)) if valid_iou else 0.0
        valid_acc = [x for x in acc_per_class if not np.isnan(x)]
        mean_acc = float(np.mean(valid_acc)) if valid_acc else 0.0

        ret = OrderedDict()
        ret["sem_seg"] = OrderedDict({
            "mIoU": miou,
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
        })
        self._logger.info(f"[SemSegEvaluator] {self._dataset_name} 分割评测结果:")
        self._logger.info(f"  mIoU: {miou:.4f}")
        self._logger.info(f"  pixel_acc: {pixel_acc:.4f}")
        self._logger.info(f"  mean_acc: {mean_acc:.4f}")
        # per-class IoU 单独打印到日志（不放进返回 dict，避免 print_csv_format
        # 对 list 做 :.4f 格式化报错；类名从 MetadataCatalog 取，若取不到用序号）
        try:
            from detectron2.data import MetadataCatalog
            cls_names = MetadataCatalog.get(self._dataset_name).thing_classes
        except Exception:
            cls_names = [str(i + 1) for i in range(self._num_classes)]
        per_cls_str = ", ".join(
            f"{cls_names[i]}={iou_per_class[i]:.4f}"
            for i in range(len(iou_per_class)) if not np.isnan(iou_per_class[i])
        )
        self._logger.info(f"  per-class IoU: {per_cls_str}")
        return ret
