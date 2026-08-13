"""
梯度裁剪（global grad-norm）监控工具
=====================================

为什么需要这个文件
------------------
日志里打印的 loss 是标量
而梯度裁剪(SOLVER.CLIP_GRADIENTS, full_model, CLIP_VALUE=1.0) 作用的对象是「所有参数 .grad 拼成的全局向量的 L2 范数」
两者量纲无关，无法从 loss 反推。

本工具在 optimizer.step() 内部、裁剪发生之前，测量一次「裁剪前的全局梯度范数 raw_total_norm」，
并统计它相对 CLIP_VALUE 的比例（clip_coef）。这样就能在训练日志里直接看到：
  - 裁剪有没有被触发
  - 触发了多大幅度（被压成了几分之一）
  - 趋势（随 iter 上升/下降）

实现方式：用闭包包裹 optimizer.step（monkey-patch），不修改 detectron2 / 项目源码，
兼容已经包了 AdamWWithGradientClip 的 optimizer（full_model 裁剪会照常执行）。

用法
----
在 main.py 的 Trainer.__init__ 末尾添加：
    self._gradnorm_stats = install_gradnorm_logger(self)

统计对象 self._gradnorm_stats 暴露：
  - .last_raw_norm      最近一次裁剪前的全局梯度范数
  - .last_clip_coef     最近一次的缩放系数（min(1.0, CLIP_VALUE/norm)）
  - .last_triggered     最近一次是否触发裁剪（norm > CLIP_VALUE）
  - .clip_value         当前裁剪阈值
  - .triggered_count    累计触发次数
  - .total_count        累计 step 次数
  - .last_contrib       最近一次各角色 {lr/n_elem/grad_norm} 字典

角色贡献说明（contrib 打印行）
------------------------------
按 backbone / proposal / head 三类分别报告：
  - n        该角色参与求范数的梯度元素总数
  - norm     该角色所有梯度拼成向量的 L2 范数
  - (x.x%)   该角色范数² / 全局范数²，即「梯度能量」占比
据此可判断 raw_norm 到底由谁主导，进而对症调参。

关于 lr 打印
-------------
遍历所有 param_group 的 lr 去重后打印。
lr 在 step 内部生效，只缩放梯度能产生的作用，不改变梯度本身。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class GradNormStats:
    """记录梯度范数裁剪的运行时统计。仅主进程打印。"""

    def __init__(self, clip_value: float, every_n: int = 20):
        self.clip_value = clip_value
        self.every_n = max(1, int(every_n))

        self.last_raw_norm: float = 0.0
        self.last_clip_coef: float = 1.0
        self.last_triggered: bool = False

        self.triggered_count: int = 0
        self.total_count: int = 0

        # 滑动窗口（最近 every_n 次的均值用）
        self._recent_norms: list[float] = []
        self._window = 100

        # 最近一次各角色的梯度贡献：{"role": {"lr":..., "n_elem":..., "grad_norm":...}}
        self.last_contrib: dict[str, dict[str, Any]] = {}

    def update(
        self,
        raw_norm: float,
        step_idx: int,
        lrs: list[float],
        contrib: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        coef = min(1.0, self.clip_value / max(raw_norm, 1e-12))
        triggered = raw_norm > self.clip_value

        self.last_raw_norm = raw_norm
        self.last_clip_coef = coef
        self.last_triggered = triggered
        self.last_contrib = contrib or {}
        self.total_count += 1
        if triggered:
            self.triggered_count += 1

        self._recent_norms.append(raw_norm)
        if len(self._recent_norms) > self._window:
            self._recent_norms.pop(0)

        # 每 every_n 步打印一次汇总
        if step_idx % self.every_n == 0:
            avg = sum(self._recent_norms) / max(1, len(self._recent_norms))
            trig_rate = self.triggered_count / max(1, self.total_count)
            tag = "CLIP" if triggered else "----"
            # 遍历所有 param_group 的 lr（去重排序，避免刷屏）。
            lrs_unique = sorted(set(lrs))
            lr_str = " ".join(f"{lr:.3e}" for lr in lrs_unique)
            logger.info(
                f"[clip info] iter={step_idx} lr=({lr_str}) "
                f"raw_norm={raw_norm:7.2f} avg{self._window}={avg:7.2f} "
                f"clip_coef={coef:6.4f} trig_rate={trig_rate*100:5.1f}% [{tag}] "
                f"threshold={self.clip_value}"
            )
            # 各角色梯度贡献（能量占比 = 角色范数² / 全局范数²）
            # contrib[role] = {"lr":..., "n_elem":..., "grad_norm":...}
            if self.last_contrib:
                order = ["backbone", "proposal", "head", "decoder", "unknown"]
                parts = []
                for role in order:
                    if role not in self.last_contrib:
                        continue
                    g = self.last_contrib[role]
                    share = (g["grad_norm"] ** 2) / max(raw_norm ** 2, 1e-12) * 100
                    parts.append(f"<{role}: param={g['n_elem']}> norm={g['grad_norm']:7.2f} ({share:.1f}%)")
                if parts:
                    logger.info(f"[contrib] {' | '.join(parts)}")


def _build_role_map_rcnn(model) -> dict[int, str]:
    """
    建立 Parameter 对象 id → 角色名 的映射，用于按角色统计梯度贡献。
    SparseRCNN：backbone / proposal / head / unknown。

    按 key 子串匹配，对齐 build_optimizer 的分组逻辑：
      - "backbone" in key       → backbone（lr=BASE_LR×BACKBONE_MULTIPLIER）
      - "init_proposal" in key  → proposal（learnable query/box embeddings）
      - "head" in key           → head   （head_series.* 等，lr=BASE_LR）
      - 其余                    → unknown

    用 id(param) 作 key：DDP 包装后 optimizer.param_groups 里的参数对象与
    model.named_parameters() 返回的是同一对象（DDP 不复制参数），id 可一一对应。
    """
    base = getattr(model, "module", model)  # 去掉 DDP / DataParallel 外壳
    role_map: dict[int, str] = {}
    for key, p in base.named_parameters(recurse=True):
        if "backbone" in key:
            role = "backbone"
        elif "init_proposal" in key:
            role = "proposal"
        elif "head" in key:
            role = "head"
        else:
            role = "unknown"
        role_map[id(p)] = role
    return role_map


def _build_role_map_seg(model) -> dict[int, str]:
    """
    SemSegMetaArch：backbone / decoder / unknown。

    SemSeg 只有 self.backbone + self.decoder(FPNDecoder)。
    注意 FPNDecoder 内部还有一个叫 self.head 的子模块，参数名形如 decoder.head.* 会含子串 "head"
    这里用 "decoder" 子串优先覆盖，避免被误判进 RCNN 风格的 head 桶
    （否则 decoder.head.* → head，而 decoder.blocks.*/fuse.* → unknown）
    """
    base = getattr(model, "module", model)
    role_map: dict[int, str] = {}
    for key, p in base.named_parameters(recurse=True):
        if "backbone" in key:
            role = "backbone"
        elif "decoder" in key:
            role = "decoder"
        else:
            role = "unknown"
        role_map[id(p)] = role
    return role_map


def _build_role_map_depth(model) -> dict[int, str]:
    """
    MonoDepthMetaArch：backbone / decoder / unknown（目前与 SemSeg 同构）。
    """
    base = getattr(model, "module", model)
    role_map: dict[int, str] = {}
    for key, p in base.named_parameters(recurse=True):
        if "backbone" in key:
            role = "backbone"
        elif "decoder" in key:
            role = "decoder"
        else:
            role = "unknown"
        role_map[id(p)] = role
    return role_map


# 按 meta-arch 类名注册表分发
# 每个 meta-arch 对应一个 role-map 构建器
# 未知模型回退到 rcnn 的通用子串映射并告警
_ROLE_MAP_BUILDERS: dict[str, Callable[[Any], dict[int, str]]] = {
    "SparseRCNN": _build_role_map_rcnn,
    "SemSegMetaArch": _build_role_map_seg,
    "MonoDepthMetaArch": _build_role_map_depth,
}


def _build_role_map(model) -> dict[int, str]:
    """
    根据 meta-arch 选择对应的角色映射；未知模型回退到通用子串映射并告警。

    用 id(param) 作 key 的理由同 _build_role_map_rcnn：DDP 不复制参数对象，
    optimizer.param_groups 里的 param 与 named_parameters() 返回的是同一对象。
    """
    base = getattr(model, "module", model)  # 去掉 DDP / DataParallel 外壳
    builder = _ROLE_MAP_BUILDERS.get(type(base).__name__)
    if builder is None:
        logger.warning(
            f"[gradclip] 未知 meta-arch '{type(base).__name__}'，"
            f"回退到通用子串映射（backbone/proposal/head/unknown），角色统计可能不准。"
        )
        return _build_role_map_rcnn(model)
    return builder(model)


def install_gradnorm_logger(trainer, every_n: int = 20) -> Optional[GradNormStats]:
    """
    包裹 trainer 持有的 optimizer.step，注入梯度范数测量。

    Args:
        trainer: DefaultTrainer 实例（或任何带 .optimizer 属性的对象）。
        every_n: 每多少步打印一次汇总日志。

    Returns:
        GradNormStats 统计对象；若无法识别 optimizer（例如被 freeze/无 grad）返回 None。
    """
    # 兼容 DefaultTrainer 把 optimizer 代理到 _trainer.optimizer 的情况
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None and getattr(trainer, "_trainer", None) is not None:
        optimizer = trainer._trainer.optimizer
    if optimizer is None:
        logger.warning("[gradclip] 找不到 optimizer，跳过梯度监控。")
        return None

    # 读取裁剪阈值；优先从 optimizer 上挂的 cfg，否则退回 detectron2 全局 CfgNode
    clip_value = 1.0
    try:
        from detectron2.config import CfgNode  # noqa: F401
        # full_model 模式阈值在 SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        cfg = getattr(trainer, "cfg", None)
        if cfg is not None and cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            clip_value = float(cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
    except Exception:
        pass

    stats = GradNormStats(clip_value=clip_value, every_n=every_n)

    # 建立 param id → role 映射（基于模型结构，不依赖分组顺序）。
    # DefaultTrainer 暴露 self.model；若不可用则 role_map 为空，只打印全局范数。
    model = getattr(trainer, "model", None)
    role_map: dict[int, str] = {}
    if model is not None:
        try:
            role_map = _build_role_map(model)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[gradclip] 无法建立角色映射，将只打印全局范数：{e}")

    original_step = optimizer.step

    def patched_step(closure=None):
        # 在真正的 step（含 full_model 裁剪）执行前，单次遍历所有参数：
        # 同时累加全局梯度平方和（global_sq）、各角色梯度平方和（contrib[role]["sq_sum"]）
        # 最后统一开方得范数
        global_sq = 0.0
        contrib: dict[str, dict[str, Any]] = {}
        lrs: list[float] = []
        has_grad = False

        for grp in optimizer.param_groups:
            lr = float(grp.get("lr", 0.0))
            lrs.append(lr)
            for p in grp["params"]:
                grad = p.grad
                if grad is None:
                    continue
                has_grad = True
                sq = float(grad.detach().float().pow(2).sum().cpu())  # 用 detach().to(cpu) 避免建立计算图 / 跨设备问题
                global_sq += sq
                if role_map:
                    role = role_map.get(id(p), "unknown")
                    c = contrib.setdefault(role, {"lr": lr, "n_elem": 0, "sq_sum": 0.0})
                    c["n_elem"] += grad.numel()
                    c["sq_sum"] += sq

        if has_grad:
            # 平方和开根号 → 全局范数 + 各角色范数
            raw_norm = global_sq ** 0.5
            for c in contrib.values():
                c["grad_norm"] = c.pop("sq_sum") ** 0.5
            # 取当前 iter（detectron2 SimpleTrainer 有 iter 属性）
            step_idx = getattr(trainer, "iter", stats.total_count)
            stats.update(raw_norm, step_idx, lrs, contrib)

        return original_step(closure)

    optimizer.step = patched_step  # type: ignore[assignment]
    logger.info(
        f"[gradclip] 已安装梯度范数监控: clip_type=full_model, CLIP_VALUE={clip_value}, "
        f"every_n={every_n}, role_map={len(role_map)} params"
    )
    return stats
