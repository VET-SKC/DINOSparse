"""DINOv3 backbone 注册层（dinosparse 框架侧的薄胶水层）。

职责：
  - 持有 third_party.dinov3.DINOv3ViT 作为底层 ViT 实现；
  - 把 wrapper 返回的 patch 特征转成 detectron2 风格 dict[str->Tensor]；
  - 实现 Backbone 抽象基类要求的 forward / output_shape / size_divisibility；
  - 用 BACKBONE_REGISTRY 注册 build 函数，供 cfg.MODEL.BACKBONE.NAME 引用。

注意：DINOv3 是单尺度 ViT，所有 transformer 层的 patch 特征都是同一个 stride（patch_size=16）
不同于 Swin 的多尺度（每个 stage stride 翻倍）
因此本 backbone 输出的多个 feature map 在空间上同尺寸、同 stride，仅通道语义不同（来自不同深度）。

权重冻结由 third_party 层的 freeze=True 控制，本层不再额外处理 requires_grad。
"""
import torch.nn as nn

from detectron2.layers import ShapeSpec

from .backbone import Backbone
from .build import BACKBONE_REGISTRY


class DINOv3Backbone(Backbone):
    """DINOv3 ViT backbone（不含 FPN）。

    输出 dict[str->Tensor]，键形如 "dinov3_stage{0,1,2}"，对应 out_indices 选定的层。
    所有输出 feature 共享同一 stride（patch_size）与通道数（embed_dim）。

    Args:
        model_path: 本地 HF 权重目录。
        out_indices: 取哪些 transformer 层（None 表示只取最后一层）。
        freeze: 是否冻结权重。
        out_feature_names: 自定义输出特征名；None 则自动按 out_indices 长度生成。
        aux_token_mode: 额外 token 选项，forward 返回 dict 包含 "aux_cls"/"aux_register"。
    """

    def __init__(self, model_path, out_indices=None, freeze=True, out_feature_names=None, aux_token_mode="none"):
        super().__init__()
        # 延迟导入，避免 dinosparse 框架顶层强依赖 transformers（仅使用 DINOv3 时才需要）
        from third_party.dinov3 import DINOv3ViT

        self.vit = DINOv3ViT(model_path=model_path, out_indices=out_indices, freeze=freeze)

        # 统一结构参数（从 wrapper 透传）
        self.embed_dim = self.vit.hidden_size
        self.patch_size = self.vit.patch_size
        self.stride = self.patch_size  # 单尺度：所有层 patch 特征的 stride 都是 patch_size
        self.freeze = freeze
        self.aux_token_mode = aux_token_mode

        n_out = len(self.vit.out_indices)
        if out_feature_names is None:
            out_feature_names = ["dinov3_stage{}".format(i) for i in range(n_out)]
        assert len(out_feature_names) == n_out, (
            f"out_feature_names 数量 {len(out_feature_names)} != out_indices 数量 {n_out}"
        )

        # Backbone 基类 output_shape() 默认实现依赖这三个属性（见 backbone.py）
        self._out_features = list(out_feature_names)
        self._out_feature_channels = {name: self.embed_dim for name in self._out_features}
        self._out_feature_strides = {name: self.stride for name in self._out_features}
        self._size_divisibility = self.patch_size

    @property
    def size_divisibility(self):
        # 覆盖基类，确保输入 H/W 被 patch_size 整除（DINOv3 的硬性要求）
        return self._size_divisibility

    def forward(self, images):
        """前向。

        Args:
            images: (B, 3, H, W) 已归一化的 RGB 张量。

        Returns:
            dict[str->Tensor]，键为输出特征名，值为 (B, embed_dim, H/patch, W/patch)。

        注：补 final LayerNorm 的逻辑在 third_party/dinov3/wrapper.py（DINOv3ViT.APPLY_FINAL_NORM），
            本层只做 dict 重组，不再二次补 norm。
        """
        out = self.vit(images)
        patch_tokens = out["patch_tokens"]  # list of (B, C, H', W'), 长度 = len(out_indices)

        features = {}
        for name, feat in zip(self._out_features, patch_tokens):
            features[name] = feat

        # 额外 token （原始维度，投影由下游 FPN 类负责）
        # aux_* 不进 _out_features / output_shape()（无 stride，非空间特征），保持 ShapeSpec 要求。
        # 只取最后一层（cls_tokens[-1] / register_tokens[-1]）。
        # TODO: 目前 token 固定取最后一层。
        #       对于来源多层的FPN，理想情况 token 层级应与 ROIPooler 分配的 FPN level 一致（小 box→浅层 token，大 box→深层 token）。
        if self.aux_token_mode != "none":
            features["aux_cls"] = out["cls_tokens"][-1]  # (B, 1, C)
            if self.aux_token_mode == "cls_register" and out["register_tokens"] is not None:
                features["aux_register"] = out["register_tokens"][-1]  # (B, num_reg, C)

        return features

    # output_shape() 沿用基类默认实现（读 _out_features/_out_feature_channels/_out_feature_strides）


@BACKBONE_REGISTRY.register()
def build_dinov3_backbone(cfg, input_shape):
    """从 config 构建纯 DINOv3 backbone（不含 FPN）。

    cfg.MODEL.DINOv3.* 字段：
        WEIGHTS         本地 HF 权重目录路径
        OUT_INDICES     取哪些 transformer 层（list[int]）；None/空 表示只取最后一层
        FREEZE          是否冻结权重（默认 True，即插即用、不微调）
        OUT_FEATURES    输出特征名（list[str]），长度需与实际输出层数一致；空则自动生成
        AUX_TOKEN_MODE  额外 token 选项 "none"/"cls"/"cls_register"

    Returns:
        DINOv3Backbone 实例。
    """
    out_indices = cfg.MODEL.DINOv3.OUT_INDICES
    out_indices = list(out_indices) if len(out_indices) > 0 else None
    out_feature_names = list(cfg.MODEL.DINOv3.OUT_FEATURES) if len(cfg.MODEL.DINOv3.OUT_FEATURES) > 0 else None

    backbone = DINOv3Backbone(
        model_path=cfg.MODEL.DINOv3.WEIGHTS,
        out_indices=out_indices,
        freeze=cfg.MODEL.DINOv3.FREEZE,
        out_feature_names=out_feature_names,
        aux_token_mode=cfg.MODEL.DINOv3.AUX_TOKEN_MODE,
    )
    # 通道数自检：与 input_shape 无关，但保留断言以防误配
    assert backbone.embed_dim == backbone._out_feature_channels[backbone._out_features[0]]
    return backbone
