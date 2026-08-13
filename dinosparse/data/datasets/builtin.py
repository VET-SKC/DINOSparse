import os
from .meta_emds7 import register_meta_emds7
from .meta_nyu import register_meta_nyu
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# ==== Predefined splits for EMDS7 ===========
def register_all_emds7(root="datasets"):
    # [name, dirname, split, prefix, annoXX]
    METASPLITS = [
        ("emds7_trainval_base_anno70", "EMDS7", "trainval", "base", 70),
        ("emds7_trainval_base_anno50", "EMDS7", "trainval", "base", 50),
        ("emds7_trainval_base_anno30", "EMDS7", "trainval", "base", 30),
        ("emds7_test_base_anno70", "EMDS7", "test", "base", 70),
        ("emds7_test_base_anno50", "EMDS7", "test", "base", 50),
        ("emds7_test_base_anno30", "EMDS7", "test", "base", 30),
        ("emds7_test_all_anno70", "EMDS7", "test", "all", 70),
        ("emds7_test_all_anno50", "EMDS7", "test", "all", 50),
        ("emds7_test_all_anno30", "EMDS7", "test", "all", 30)
    ]
    # shots
    for prefix in ["all"]:
        for annoXX in [70, 50, 30]:
            for shot in [1, 2, 3, 5, 8]:
                for seed in range(10):
                    seed = "" if seed == 0 else "_seed{}".format(seed)
                    name = "emds7_trainval_{}_anno{}_{}shot{}".format(prefix, annoXX, shot, seed)  # 数据集的注册名称
                    dirname = "EMDS7"
                    split = "trainval"
                    METASPLITS.append(
                        (name, dirname, split, prefix, annoXX)
                    )
    # do register
    for name, dirname, split, prefix, annoXX in METASPLITS:
        register_meta_emds7(
            name,                                       # name
            _get_builtin_metadata("emds7_fewshot"),     # metadata_classes
            os.path.join(root, dirname),                # dirname
            split,                                      # split
            prefix,                                     # prefix
            annoXX                                      # annoXX
        )


def register_all_nyu(root="datasets"):
    """
    NYU Depth V2 数据集注册入口。

    定义 4 个 split 并注册（train/test × depth/seg）：
      nyu_depth_train (795)  nyu_depth_test  (654)
      nyu_seg_train   (795)  nyu_seg_test    (654)

    RGB 与深度和分割 GT 共用同一批图（导出时按 splits.mat 的 795/654 划分），
    两个任务只是读不同的 *_file_name 字段，底层 PNG 目录相同。
    """
    # [name, split, task]
    SPLITS = [
        ("nyu_depth_train", "train", "depth"),
        ("nyu_depth_test", "test", "depth"),
        ("nyu_seg_train", "train", "seg"),
        ("nyu_seg_test", "test", "seg"),
    ]
    metadata = _get_builtin_metadata("nyu_seg")  # depth/seg 共用同一份类别元数据
    dirname = os.path.join(root, "NYUDepthV2")

    for name, split, task in SPLITS:
        register_meta_nyu(
            name,
            metadata,
            dirname,
            split,
            task,
        )


register_all_emds7()
register_all_nyu()
