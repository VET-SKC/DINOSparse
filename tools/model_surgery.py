"""
============================================================================
背景：DINOSparse 的 few-shot 微调流程
============================================================================
本项目 SparseR-CNN 检测器的参数按功能分 ABCD 四组（详见 dinosparse/config/defaults.py
与 dinosparse/modeling/meta_arch/rcnn.py 的 _apply_freeze）：

  A 组  init_proposal_boxes / init_proposal_features    object queries
  B 组  head.head_series.{i}.class_logits               分类头（依赖 num_classes）
        head.head_series.{i}.bboxes_delta               回归头（类别无关，固定 4 维）
  C 组  head.head_series.{i}.cls_module / reg_module    cls/reg 适配 MLP
  D 组  head.head_series.{i}.self_attn / inst_interact /
        linear1 / linear2 / norm1 / norm2 / norm3       特征交互主干

few-shot 微调时（BACKBONE.FREEZE=True + FREEZE_INTERACTION=True）：
backbone（DINOv3ViT+FPN）与 D 组冻结，A/B/C 组可训练。

============================================================================
本脚本只改 B 组的分类头，其余全部交给 detectron2 checkpointer 自动继承
============================================================================
                  依赖 num_classes   |   处理方式
  ─────────────────────────────────────────────────────────────────
  A 组              否                  checkpointer 自动继承
  B 组 bboxes_delta 否（固定 4 维）      checkpointer 自动继承
  B 组 class_logits 是（本脚本处理）     ★ 脚本扩充 + 基类权重映射 + 新类随机
  C 组              否                  checkpointer 自动继承
  D 组              否                  checkpointer 自动继承

bboxes_delta 是类别无关的（输出 4 维 dx,dy,dw,dh），base 与 all 维度相同，
detectron2 DetectionCheckpointer 会按形状匹配自动加载，无需手动处理。

只有 class_logits 依赖 num_classes：6 个级联 RCNNHead 各有一个
  focal   模式：nn.Linear(d_model, num_classes)          无背景类
  softmax 模式：nn.Linear(d_model, num_classes + 1)      末位为背景类
本脚本把 base 的 class_logits 扩充到 all 的类别数：
已训练的基类权重按类别映射移植（IDMAP），新增的新类权重随机初始化。

============================================================================
注意：类别索引映射（IDMAP）
============================================================================
SparseR-CNN 的 class_logits 输出张量第 i 维 ⟺ 数据集 thing_classes 列表的第 i 个元素
category_id = thing_classes.index(类名)（见 meta_emds7.py 的标注加载）
thing_classes 来自 dinosparse/data/datasets/builtin_meta.py 里硬编码的 list

emds7 的 BASE 是从升序全集里跳选的子集（非连续前 N 个），例如 anno50：
  BASE = [G001, G003, G004, G005, G008, ...]   共 25 类
  ALL  = [G001, G002, G003, G004, G005, ...]   共 41 类
base checkpoint 里 class_logits 第 0 维=G001、第 1 维=G003、第 2 维=G004...
all 模型里          class_logits 第 0 维=G001、第 1 维=G002、第 2 维=G003...
所以 base 的第 b 维要搬到 all 的第 ALL.index(BASE[b]) 维：
  IDMAP[base_idx] = ALL[anno].index(BASE[anno][base_idx])
本脚本直接 import builtin_meta.py 取 list 构造 IDMAP，与训练时的索引定义同源。

============================================================================
用法示例
============================================================================
python tools/model_surgery.py \
    --dataset emds7 --anno 50 \
    --src-path checkpoints/emds7/dinov3_vitb_fpn_sparse_base_anno50/model_0009999.pth \
    --save-dir checkpoints/emds7/dinov3_vitb_fpn_sparse_base_anno50/

python main.py \
    --config-file configs/emds7/dinov3_vitb_fpn_sparse_all_anno50_1shot_seed1.yaml \
    --opts MODEL.WEIGHTS checkpoints/emds7/dinov3_vitb_fpn_sparse_base_anno50/model_reset_surgery.pth

# --dry-run 只打印变换报告不写盘，用于验证路径推导与 IDMAP：
python tools/model_surgery.py ... --dry-run
"""
import argparse
import importlib.util
import math
import os

import torch


# ----------------------------------------------------------------------------
# builtin_meta.py 是零依赖的纯数据文件，用 importlib 按文件路径直接加载，
# 绕开 dinosparse.data.__init__ 对 detectron2 的 import 副作用。
# ----------------------------------------------------------------------------
def _load_builtin_meta(proj_root):
    meta_path = os.path.join(proj_root, "dinosparse", "data", "datasets", "builtin_meta.py")
    spec = importlib.util.spec_from_file_location("_builtin_meta", meta_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_emds7_idmap(builtin_meta, anno):
    """构造 emds7 的 base_idx -> all_idx 映射。

    base checkpoint 的 class_logits 第 b 维对应类目 BASE[anno][b]，
    它要搬到 all 模型的第 ALL[anno].index(BASE[anno][b]) 维。
    返回 dict{base_idx: all_idx}。
    """
    BASE = builtin_meta.EMDS7_BASE_CATEGORIES[anno]
    ALL = builtin_meta.EMDS7_ALL_CATEGORIES[anno]
    missing = [c for c in BASE if c not in ALL]
    assert not missing, f"BASE 类目不在 ALL 中: {missing}"
    return {b: ALL.index(BASE[b]) for b in range(len(BASE))}


# ----------------------------------------------------------------------------
# 类别数与 IDMAP。emds7 从 builtin_meta 取（与训练同源，不硬编码数字）
# coco 沿用原脚本自包含的硬编码（项目内无 COCO 数据集注册代码）
# voc 类别连续，IDMAP = identity（顺序前移）
# TODO(coco/voc): 项目当前无 COCO/VOC 数据集注册代码，下列划分未经项目内验证，仅供迁移参考；
#                 若 focal/softmax 背景类处理有疑问需在真实数据上复核。
# ----------------------------------------------------------------------------
def get_class_layout(dataset, anno, builtin_meta):
    """返回 (base_classes_num, all_classes_num, idmap, tar_total)。

    idmap: dict{base_idx: all_idx}，base 第 base_idx 维 -> all 第 all_idx 维。
    tar_total: 目标 class_logits 输出维度（不含背景）。
               对 focal = all_num，
               softmax 时背景类在调用处单独处理（本函数只返回不含背景的类别数）。
    """
    if dataset == "emds7":
        base_num = len(builtin_meta.EMDS7_BASE_CATEGORIES[anno])
        all_num = len(builtin_meta.EMDS7_ALL_CATEGORIES[anno])
        idmap = build_emds7_idmap(builtin_meta, anno)
        return base_num, all_num, idmap, all_num
    elif dataset == "coco":
        # 80 类
        # NOVEL/BASE 是 COCO 连续 ID，ALL 排序后按 ID 升序；IDMAP 把 base 在 all 中的位置算出。
        NOVEL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        BASE_CLASSES = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        # base checkpoint 的 class_logits 第 i 维对应 BASE_CLASSES[i]（base 训练时按此 list）；
        # all 模型按 ALL_CLASSES 升序为索引。IDMAP[base_idx] = ALL.index(BASE[base_idx])。
        idmap = {b: ALL_CLASSES.index(BASE_CLASSES[b]) for b in range(len(BASE_CLASSES))}
        # TODO(coco): 上述 BASE->ALL 索引假设需在真实 COCO base/all 训练上复核。
        return len(BASE_CLASSES), len(ALL_CLASSES), idmap, len(ALL_CLASSES)
    elif dataset == "voc":
        # VOC 20 类，base/novel 划分连续前移，IDMAP = identity。
        # TODO(voc): 项目内无 VOC 划分定义；若采用非连续划分需在此补充。
        all_num = 20
        # 常见 TFA voc 划分：base 15 类 + novel 5 类。此处给最小可用骨架。
        base_num = 15
        idmap = {b: b for b in range(base_num)}  # 顺序前移
        return base_num, all_num, idmap, all_num
    else:
        raise NotImplementedError(f"未知 dataset: {dataset}")


def infer_focal(prev_dim, base_num):
    """根据旧 class_logits 输出维度推断 focal/softmax。

    focal:   输出 num_classes（无背景）   -> prev_dim == base_num
    softmax: 输出 num_classes + 1（背景） -> prev_dim == base_num + 1
    """
    if prev_dim == base_num:
        return True
    elif prev_dim == base_num + 1:
        return False
    else:
        raise ValueError(
            f"旧 class_logits 输出维度 {prev_dim} 与 base 类数 {base_num} 不符"
            f"（应为 {base_num}(focal) 或 {base_num+1}(softmax)），可能 src checkpoint 不是 base 训练结果。"
        )


def expand_class_logits(ckpt, args, base_num, idmap, tar_num, prior_prob, builtin_meta):
    """对 head.head_series.{0..N-1}.class_logits 做扩充+映射+随机初始化。

    focal 模式：目标维度 tar_num，基类按 idmap 搬，其余随机。
    softmax 模式：目标维度 tar_num+1，基类按 idmap 搬、末位背景类单独保留。
    """
    n_processed = 0
    n_missing = 0
    bias_value = -math.log((1 - prior_prob) / prior_prob)  # focal 风格 bias，与 head.py _reset_parameters 一致

    for i in range(args.num_heads):
        prefix = f"head.head_series.{i}.class_logits"
        w_key, b_key = prefix + ".weight", prefix + ".bias"
        if w_key not in ckpt["model"]:
            print(f"  [warn] {w_key} 不在 checkpoint，跳过该层")
            n_missing += 1
            continue
        old_w = ckpt["model"][w_key]
        old_b = ckpt["model"].get(b_key)
        prev_dim = old_w.size(0)
        feat_size = old_w.size(1)
        use_focal = infer_focal(prev_dim, base_num)
        tar_dim = tar_num if use_focal else tar_num + 1

        # 新权重：整体 normal(0, 0.01)，bias 先全置 focal bias（新类即此值）。
        new_w = torch.randn(tar_dim, feat_size) * 0.01
        new_b = torch.full((tar_dim,), bias_value, dtype=old_w.dtype)
        # 与原脚本一致：把基类权重从旧位置搬到 idmap 指定的新位置。
        # base checkpoint 第 b 维 -> all 模型第 idmap[b] 维。
        for b_idx, a_idx in idmap.items():
            new_w[a_idx] = old_w[b_idx]
            if old_b is not None:
                new_b[a_idx] = old_b[b_idx]
        # softmax 模式：末位背景类保留旧值（旧 ckpt 末位即背景）。
        if not use_focal and old_b is not None:
            new_b[-1] = old_b[-1]
            new_w[-1] = old_w[-1]

        n_kept = len(idmap)
        n_rand = tar_dim - n_kept - (0 if use_focal else 1)
        print(f"  [{prefix}] use_focal={use_focal} | weight {tuple(old_w.shape)} -> ({tar_dim}, {feat_size})"
              f" | 基类保留 {n_kept} | 新类随机 {n_rand}"
              f"{'' if use_focal else ' | 背景类保留末位'}")

        if not args.dry_run:
            ckpt["model"][w_key] = new_w
            if old_b is not None:
                ckpt["model"][b_key] = new_b
        n_processed += 1

    return n_processed, n_missing


def main(args):
    ckpt = torch.load(args.src_path, map_location="cpu")
    assert isinstance(ckpt, dict), f"checkpoint 应为 dict，得到 {type(ckpt)}"
    assert "model" in ckpt, "checkpoint 缺少 'model' 字段"

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builtin_meta = _load_builtin_meta(proj_root)

    base_num, all_num, idmap, tar_num = get_class_layout(args.dataset, args.anno, builtin_meta)
    print(f"数据集={args.dataset} anno={args.anno} | base 类数={base_num} -> all 类数={all_num}"
          f" | IDMAP 前5项={list(idmap.items())[:5]}")
    print(f"target class_logits 输出维度（不含背景）={tar_num}, prior_prob={args.prior_prob}")

    # 清掉 scheduler/optimizer，重置 iteration，保证 few-shot 从 iter 0 起训且不沿用旧优化器状态。
    for k in ("scheduler", "optimizer"):
        if k in ckpt:
            del ckpt[k]
            print(f"已删除 checkpoint 中的 '{k}'")
    if "iteration" in ckpt:
        ckpt["iteration"] = 0

    print("\n扩充 class_logits：")
    n_proc, n_miss = expand_class_logits(
        ckpt, args, base_num, idmap, tar_num, args.prior_prob, builtin_meta
    )

    if args.dry_run:
        print(f"\n[dry-run] 处理 {n_proc} 层（跳过 {n_miss} 层），未写盘。")
        return

    save_name = args.tar_name + ".pth"
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"\n处理 {n_proc} 层（跳过 {n_miss} 层），已保存到 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对 SparseR-CNN base checkpoint model surgery（few-shot 微调用）。")
    parser.add_argument("--dataset", type=str, default="emds7", choices=["voc", "coco", "emds7"])
    parser.add_argument("--src-path", type=str, required=True, help="基类预训练 checkpoint 路径")
    parser.add_argument("--save-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--tar-name", type=str, default="model_reset_surgery", help="输出文件名（不含扩展名）")
    parser.add_argument("--num-heads", type=int, default=6, help="SparseRCNN NUM_HEADS（级联层数）")
    parser.add_argument("--anno", type=int, default=50, choices=[70, 50, 30])
    parser.add_argument("--prior-prob", type=float, default=0.01,
                        help="新类 focal bias = -log((1-p)/p)，须与 MODEL.SparseRCNN.PRIOR_PROB 一致")
    parser.add_argument("--dry-run", action="store_true", help="只打印变换报告，不写盘")
    args = parser.parse_args()
    main(args)
