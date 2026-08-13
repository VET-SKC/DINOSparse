# -*- coding: utf-8 -*-
"""
NYU Depth V2 labeled 子集的数据加载与注册。

数据来源（已由 export_to_png.py 导出成 PNG 目录，详见该脚本头部说明）：
  datasets/NYUDepthV2/images/{train,test}/000xxx.png    RGB
  datasets/NYUDepthV2/depth/{train,test}/000xxx.png     16-bit PNG，depth_mm
  datasets/NYUDepthV2/labels40/{train,test}/000xxx.png  8-bit PNG，0=void，1–40=类

注册到 detectron2 的 DatasetCatalog / MetadataCatalog

每条 dataset_dict:
  - 通用: file_name / image_id / height / width
  - 深度任务: 加 "depth_file_name"（由 DepthDatasetMapper 读取）
  - 分割任务: 加 "sem_seg_file_name"（由默认 DatasetMapper 读取，原生支持）
注意：NYU 是密集预测，无 bbox/instances 标注，故 annotations 字段为空。
"""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.file_io import PathManager

__all__ = ["load_nyu_instances", "register_meta_nyu"]


def load_nyu_instances(dirname: str, split: str, task: str):
    """
    遍历某 split 的 PNG 目录，生成 detectron2 dataset_dicts。

    Args:
        dirname: NYUDepthV2 根目录（含 images/depth/labels40 子目录）
        split: "train" / "test"
        task: "depth" / "seg"，决定加哪个 *_file_name 字段
    """
    from PIL import Image

    img_dir = os.path.join(dirname, "images", split)
    # 连续编号 000000.png ...
    names = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    assert task in ("depth", "seg"), f"unknown task {task}"

    dataset_dicts = []
    for i, name in enumerate(names):
        record = {
            "file_name": os.path.join(img_dir, name),
            "image_id": i,
            "height": 480,
            "width": 640,
        }
        if task == "depth":
            record["depth_file_name"] = os.path.join(dirname, "depth", split, name)
        else:  # seg
            record["sem_seg_file_name"] = os.path.join(dirname, "labels40", split, name)
        dataset_dicts.append(record)

    return dataset_dicts


def register_meta_nyu(name: str, metadata_classes, dirname: str, split: str, task: str):
    """
    注册一个 NYU 数据集 split 到 DatasetCatalog / MetadataCatalog。

    Args:
        name: 注册名，如 "nyu_depth_train"
        metadata_classes: _get_builtin_metadata(...) 返回的字典
        dirname: NYUDepthV2 根目录
        split: "train" / "test"
        task: "depth" / "seg"，决定 evaluator_type 和加载哪个字段
    """
    if task == "depth":
        evaluator_type = "nyu_depth"
    else:  # seg
        evaluator_type = "nyu_seg"

    thing_classes = metadata_classes["thing_classes"]

    DatasetCatalog.register(
        name,
        lambda: load_nyu_instances(dirname, split, task),
    )
    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        stuff_classes=thing_classes,            # 语义分割本质是 stuff，同时塞一份
        evaluator_type=evaluator_type,
        dirname=dirname,
        split=split,
        task=task,
        ignore_label=0,                         # labels40 的 0=void（分割评测用） 这个字段目前未使用
    )
