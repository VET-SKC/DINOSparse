"""DINOv3 ViT 封装层

设计原则：
  - 纯 torch + transformers 实现，可独立测试、独立复用。
  - 输入张量应是已归一化的 RGB 图像 (B, 3, H, W)。
    在 detectron2 中，preprocess_image 已用 MODEL.PIXEL_MEAN/STD（RGB ImageNet）完成归一化，
    本模块不再重复归一化，直接传入 ViT。
  - H, W 必须是 patch_size(默认16) 的整数倍。
  - 权重冻结：freeze=True 时不参与训练（requires_grad=False + eval + no_grad）。

输出结构见 tools/verify_dinov3.py（DINOv3ViTModel 输出 BaseModelOutputWithPooling）：
  token 序列顺序 = CLS(0) → register(1..num_reg) → patch(num_reg+1 ..)，
  last_hidden_state 已经过顶层 final LayerNorm。
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DINOv3ViT(nn.Module):
    """HuggingFace DINOv3ViT 的轻量封装，提供统一的特征提取接口。

    Args:
        model_path: 本地权重目录路径（含 config.json + model.safetensors）。
        out_indices: 要返回哪些 transformer 层的 patch 特征。
            transformers 的 hidden_states 是 (num_hidden_layers + 1) 元组，
            index 0 是 embedding 输出，index 1..N 是第 1..N 个 transformer 层输出。
            传 None 表示只返回最后一层（last_hidden_state）。
            例：[9, 11, 12] 返回第9层、第11层、第12层（最后一层）。
        freeze: True 则冻结权重并设为 eval 模式，forward 在 no_grad 下进行。
    """

    # === 是否对中间层补 final LayerNorm（显式设定，直接改这里切换，不走 config）===
    # 源码已证实 (transformers/models/dinov3_vit/modeling_dinov3_vit.py)：
    # HF 的 hidden_states[0..N] 全部未过模型级 final LayerNorm (self.model.norm)，
    # HF 的 last_hidden_state = self.norm(hidden_states[N])，过了 norm。（tie_last_hidden_states=False）。
    #   - 装饰器 @capture_outputs(tie_last_hidden_states=False) 明确不让两者一致。
    # 单层模式（out_indices=[最后一层]）用 last_hidden_state（已 norm）；
    # 多层模式（其余）用 hidden_states（未 norm）。
    # 两种模式下元素的 norm 状态不同。
    #
    # 设 True：对未过 norm 的层补 final_norm，使所有层与 last_hidden_state 尺度一致。
    #   - 补 norm 在 token 序列（BLC）阶段进行，cls/register/patch 三类 token 共享同一 normed 序列。
    # 设 False：保留 HF 原始输出（多层未 norm、单层已 norm，混用需谨慎）。
    APPLY_FINAL_NORM = True

    def __init__(self, model_path: str, out_indices: Optional[List[int]] = None, freeze: bool = True):
        super().__init__()
        assert os.path.isdir(model_path), f"权重目录不存在: {model_path}"

        self.model_path = model_path
        self.freeze = freeze
        self.config = AutoConfig.from_pretrained(self.model_path)

        # 关键结构参数（预留，供读取）
        self.hidden_size = getattr(self.config, "hidden_size", 768)  # embed_dim
        self.num_channels = getattr(self.config, "num_channels", 3)
        self.num_attention_heads = getattr(self.config, "num_attention_heads", 12)
        self.num_hidden_layers = getattr(self.config, "num_hidden_layers", 12)
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 4)
        self.patch_size = getattr(self.config, "patch_size", 16)

        # token 序列中 patch 部分的起始下标 = 1(CLS) + num_register_tokens
        self._patch_offset = 1 + self.num_register_tokens

        # 决定要取哪些层。None → 只取最后一层（用 last_hidden_state）。
        if out_indices is None:
            self.out_indices = [self.num_hidden_layers]  # 最后一层
        else:
            # 规范化：去重排序，范围检查
            self.out_indices = sorted(set(out_indices))
            for i in self.out_indices:
                assert 0 <= i <= self.num_hidden_layers, f"out_indices 超出范围 [0, {self.num_hidden_layers}]: {i}"

        # 是否需要中间层 → 决定 forward 时是否开 output_hidden_states。
        self.need_hidden_states = self.out_indices != [self.num_hidden_layers]

        self.model = AutoModel.from_pretrained(self.model_path)

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    # ---------------------------------------------------------------
    # 对外属性
    # ---------------------------------------------------------------
    @property
    def size_divisibility(self) -> int:
        """输入 H、W 需被 patch_size 整除。"""
        return self.patch_size

    def get_patch_feature_map(self, hidden: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """把某一层的 token 序列 (B, L, C) 还原为空间特征图 (B, C, H', W')。

        Args:
            hidden: (B, L, C) 的 token 序列（含 CLS/register）。
            H, W: 原始输入的高宽（用于算 H'=H/patch, W'=W/patch）。
        """
        B, L, C = hidden.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        patch_tokens = hidden[:, self._patch_offset:, :]        # (B, Hp*Wp, C)
        assert patch_tokens.shape[1] == Hp * Wp, (
            f"patch token 数 {patch_tokens.shape[1]} != H'*W'={Hp*Wp}，"
            f"可能是 H/W({H}/{W}) 非 patch_size({self.patch_size}) 整数倍"
        )
        return patch_tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()

    # ---------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> dict:
        """前向，提取多层的三类 token（cls / register / patch），均以 list 形式返回。

        transformers/models/dinov3_vit/modeling_dinov3_vit.py
            - Encoder.forward (line 500-503)：
                逐层计算，返回 last_hidden_state=最后一层输出，但 encoder 内部无 final LayerNorm。
            - DINOv3ViTModel.forward (line 539-544)：
                output = self.model(hidden_states, position_embeddings, **kwargs)
                sequence_output = self.norm(output.last_hidden_state)   # final LayerNorm 只在此应用一次
                pooled_output = sequence_output[:, 0, :]

                return BaseModelOutputWithPooling(
                    last_hidden_state=sequence_output,                  # 过了 norm
                    pooler_output=pooled_output,
                    hidden_states=output.hidden_states,                 # 直接转发 encoder，未过 norm
                    attentions=output.attentions,
                )
            - 装饰器 @capture_outputs(tie_last_hidden_states=False) (line 493)
                明确不把last_hidden_state 绑定回 hidden_states[-1]，故两者数值不同。
            - 结论：hidden_states[0..N] 全部未过 final norm；last_hidden_state 过了。
                torch.all(model.norm(hs[-1]) == lhs)
                tensor(True)

        单层模式（out_indices=[最后一层]）：用 last_hidden_state（已过 norm），list 只有1个元素。
        多层模式（out_indices 含其他层）：用 hidden_states（全未过 norm），list 含多个元素。
        ⚠ 两种模式下元素的 norm 状态不同，混用时务必在 dinov3.py 统一补 norm。

        Args:
            images: (B, 3, H, W)，已归一化的 RGB 张量，H/W 为 patch_size 整数倍。

        Returns:
            dict:
              "patch_tokens":    list[ (B, C, H', W') ]，每层一个（已 reshape 成空间图）
              "cls_tokens":      list[ (B, 1, C) ]，每层一个（保持 BLC 维度）
              "register_tokens": list[ (B, num_reg, C) ] 或 None（num_reg=0 时）
              "out_indices":     list[int]，实际取的 transformer 层号
              "H", "W":          原始输入尺寸
        """
        B, C, H, W = images.shape
        assert C == self.num_channels, f"输入通道 {C} != 模型通道 {self.num_channels}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"输入尺寸 H/W={H}/{W} 必须是 patch_size={self.patch_size} 的整数倍"
        )

        ctx = torch.no_grad() if (self.freeze and not self.training) else _identity_ctx()
        with ctx:
            out = self.model(images,
                             output_hidden_states=self.need_hidden_states,  # .hidden_states
                             output_attentions=False)  # .attentions （未使用）
            # .last_hidden_state    [batch_size, num_tokens, embed_dim]
            # .pooler_output        [batch_size, embed_dim] （未使用）

        # 取出各指定层的完整 token 序列 (B, L, C)
        if self.need_hidden_states:
            hs = out.hidden_states  # tuple, len = num_layers+1
            layer_tokens = [hs[i] for i in self.out_indices]
        else:
            # 单层且为最后一层：用 last_hidden_state（注意它已过 final norm，与多层模式的 norm 状态不同）
            layer_tokens = [out.last_hidden_state]

        # 集中补 norm（在 BLC 阶段，对整个 token 序列补 final LayerNorm）
        # 单层模式（out_indices=[最后一层]）用 last_hidden_state（已过 norm），
        # 多层模式用 hidden_states（未过 norm）。
        #       - need_hidden_states=True（多层）→ 全部未过 norm，补
        #       - need_hidden_states=False（单层）→ 已过 norm，不补
        # 补 norm 后 cls/register/patch 三类 token 共享同一 normed 序列，与 last_hidden_state 尺度一致。
        if self.APPLY_FINAL_NORM and self.need_hidden_states:
            final_norm = self.model.norm  # final LayerNorm（权重已含）
            layer_tokens = [final_norm(tok) for tok in layer_tokens]

        # 拆分三类 token，每层都产出，按层排列成 list
        patch_tokens: List[torch.Tensor] = []
        cls_tokens: List[torch.Tensor] = []
        register_tokens: List[torch.Tensor] = []
        for tok in layer_tokens:
            # patch → 空间特征图 (B, C, H', W')
            patch_tokens.append(self.get_patch_feature_map(tok, H, W))
            # cls → (B, 1, C) 保持 BLC 维度
            cls_tokens.append(tok[:, 0:1, :])
            # register → (B, num_reg, C)
            if self.num_register_tokens > 0:
                register_tokens.append(tok[:, 1:1 + self.num_register_tokens, :])

        return {
            "patch_tokens": patch_tokens,
            "cls_tokens": cls_tokens,
            "register_tokens": register_tokens if self.num_register_tokens > 0 else None,
            "out_indices": list(self.out_indices),
            "H": H,
            "W": W,
        }


class _identity_ctx:
    """torch.no_grad() 的 no-op 替代，用于 freeze=False 时保持 with 语法统一。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
