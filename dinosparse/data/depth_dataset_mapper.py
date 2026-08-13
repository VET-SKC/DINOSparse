# -*- coding: utf-8 -*-
"""
单目深度估计专用的 DatasetMapper。

为什么需要单独的 mapper：
  detectron2 的默认 DatasetMapper 已原生支持 sem_seg_file_name
  （语义分割 GT 会随 image 的 flip/resize 同步变换）
  所以分割任务直接用默认 mapper
  但深度图是 float 回归 GT，AugInput 不认 depth 字段，d2 的增强管线不会自动同步它
  因此这里读出 depth，先对 image 跑增强拿到 transforms，
  再用 transforms.apply_image(depth) 手动把同一组几何变换（flip/resize）同步应用到深度图上

实现策略：继承现有 DatasetMapper，只重写 __call__，复用父类的 augmentation 构建逻辑，
保证和检测/分割管线走完全相同的增强（ResizeShortestEdge + RandomFlip）。
"""
import copy

import numpy as np
import torch
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .dataset_mapper import DatasetMapper


class DepthDatasetMapper(DatasetMapper):
    """单目深度估计的 dataset mapper。

    与父类 DatasetMapper 的区别：
      - 额外读取 dataset_dict["depth_file_name"]（16-bit PNG，depth_mm），还原成 float32 米；
      - 对 image 跑增强后，用同一组 transforms 同步变换 depth；
      - 输出 dict 加 "depth"（HxW float32 Tensor），并丢弃检测专用的 annotations/instances。
    """

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # 1. 读 RGB
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # 2. 读深度（16-bit PNG depth_mm -> float32 米）
        if "depth_file_name" not in dataset_dict:
            raise ValueError(
                "DepthDatasetMapper 需要 dataset_dict['depth_file_name']，"
                "请确认注册时 task='depth'。"
            )
        depth_mm = np.asarray(Image.open(dataset_dict["depth_file_name"]), dtype=np.float32)
        depth_m = depth_mm / 1000.0  # depth_mm -> 米

        # 3. 对 image 跑增强，拿到 transforms（标准 d2 流程）
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # 4. 用同一组 transforms 同步变换 depth（关键）
        depth_m = transforms.apply_image(depth_m)

        # 5. 装箱成 Tensor
        # RGB: HWC -> CHW
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # depth: HxW float32
        dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth_m.astype(np.float32)))

        # 6. 清理检测专用字段（深度任务无 bbox/instances/sem_seg）
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        dataset_dict.pop("depth_file_name", None)

        return dataset_dict
