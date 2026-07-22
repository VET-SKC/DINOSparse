import os
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

__all__ = ["register_meta_emds7"]


def load_filtered_emds7_instances(name: str, dirname: str, split: str, classes: str):
    """
    Load EMDS7 detection annotations to Detectron2 format.
    Args:
        dirname: Contain "EMDS7Images", "EMDS7Sets", "EMDS7Xml"
        split (str): one of "trainval", "test"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "emds7split")
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classes:
            with PathManager.open(os.path.join(split_dir, "box_{}shot_{}_train.txt".format(shot, cls))) as f:
                file_list_per_txt = np.loadtxt(f, dtype=str).tolist()
                if isinstance(file_list_per_txt, str):
                    file_list_per_txt = [file_list_per_txt]
                fileid_per_cls = [fid.split("/")[-1].split(".png")[0] for fid in file_list_per_txt]  # 获取主名
                fileids[cls] = fileid_per_cls
    else:
        with PathManager.open(os.path.join(dirname, "EMDS7Sets", split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=str)

    # 小样本时，fileids是dict{"class": list[str]}      # 不带扩展名
    # 大样本时，fileids是ndarray类似list[str]           # 带扩展名

    dicts = []
    if is_shots:
        for cls, fileids_per_cls in fileids.items():
            dicts_ = []
            for fileid in fileids_per_cls:
                anno_file = os.path.join(dirname, "EMDS7Xml", fileid + ".xml")
                png_file = os.path.join(dirname, "EMDS7Images", split, fileid + ".png")

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    # 小样本，一个实例一个dict，annotations字段的list长度为1，一个图片文件可能分几个dict表示
                    single_dict = {
                        "file_name": png_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text)
                    }
                    obj_cls = obj.find("name").text
                    if cls != obj_cls:
                        continue
                    # 找到该类实例后

                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    annotations = [
                        {
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": classes.index(cls)
                        }
                    ]

                    single_dict["annotations"] = annotations

                    # dicts_在内循环结束后保存的是该类全部实例
                    # dicts_在外循环结束后保存的是所有类全部实例
                    dicts_.append(single_dict)

            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            fileid = fileid.split(".png")[0]
            anno_file = os.path.join(dirname, "EMDS7Xml", fileid + ".xml")
            png_file = os.path.join(dirname, "EMDS7Images", split, fileid + ".png")

            tree = ET.parse(anno_file)

            # 大样本，一个图片文件一个dict，annotations字段的list长度为非负整数
            single_dict = {
                "file_name": png_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }

            annotations = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classes):
                    continue
                # 找到所需类别范围内的实例后

                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                annotations.append(
                    {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": classes.index(cls)
                    }
                )

            single_dict["annotations"] = annotations

            # 循环结束后保存的是所有文件的dict
            dicts.append(single_dict)

    return dicts


def register_meta_emds7(name, metadata_classes, dirname, split, prefix, annoXX):
    if prefix.startswith("all"):
        thing_classes = metadata_classes["EMDS7_ALL_CATEGORIES"][annoXX]
    elif prefix.startswith("base"):
        thing_classes = metadata_classes["EMDS7_BASE_CATEGORIES"][annoXX]
    elif prefix.startswith("novel"):
        thing_classes = metadata_classes["EMDS7_NOVEL_CATEGORIES"][annoXX]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_emds7_instances(name, dirname, split, thing_classes),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        split=split,
        base_classes=metadata_classes["EMDS7_BASE_CATEGORIES"][annoXX],
        novel_classes=metadata_classes["EMDS7_NOVEL_CATEGORIES"][annoXX],
        evaluator_type="emds7"
    )
