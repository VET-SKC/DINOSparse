from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C

# -------- EMDS7 BGR mean std, 0-255 -------- #
_CC.MODEL.PIXEL_MEAN = [86.147, 173.193, 116.081]
_CC.MODEL.PIXEL_STD = [20.435, 37.453, 24.487]
_CC.MODEL.YCRCB_MEAN = [146.184, 106.498, 94.133]
_CC.MODEL.YCRCB_STD = [30.517, 8.249, 10.823]

# ------------- DINOv3 ------------- #
# DINOv3 ViT 主干（基于 HuggingFace transformers，权重冻结、即插即用、不微调）。
_CC.MODEL.DINOv3 = CN()
_CC.MODEL.DINOv3.WEIGHTS = ""                                  # 本地 HF 权重目录（含 config.json + model.safetensors）
_CC.MODEL.DINOv3.OUT_INDICES = ()                              # 取哪些 transformer 层；空() 表示取最后一层
_CC.MODEL.DINOv3.OUT_FEATURES = []                             # 输出特征名；空[] 表示自动生成 dinov3_stage{0..n}
_CC.MODEL.DINOv3.FEATURE_STRATEGY = "simple"                   # FPN 构造方式
_CC.MODEL.DINOv3.PATCH_SIZE = 16                               # DINOv3 patch_size（信息字段，实际从权重 config 读）
_CC.MODEL.DINOv3.FREEZE = True                                 # 冻结权重（即插即用，默认 True）
# 分层冻结：
# -- MODEL.DINOv3.FREEZE      只冻 DINOv3ViT（在 DINOv3Backbone 内部处理，始终为 True）
# -- MODEL.BACKBONE.FREEZE    连同 FPN 一起冻结（few-shot 微调阶段用）

# ------------- SparseRCNN ------------- #
# SparseR-CNN 端到端检测头（无 RPN，learnable object queries + DynamicHead）。
# 整合自 https://github.com/PeizeSun/SparseR-CNN，默认值与原版一致。
# 注意：该头不使用 RPN，也不走 ROI_HEADS；NUM_CLASSES 应与数据集类别数一致。
_CC.MODEL.SparseRCNN = CN()
_CC.MODEL.SparseRCNN.NUM_CLASSES = 80              # 类别数（不含背景，focal 模式）；yaml 里按数据集覆盖
_CC.MODEL.SparseRCNN.NUM_PROPOSALS = 100           # 可学习的 proposal（object query）数量

_CC.MODEL.SparseRCNN.NHEADS = 8                    # MultiheadAttention 头数
_CC.MODEL.SparseRCNN.NUM_DYNAMIC = 2               # DynamicConv 参数组数
_CC.MODEL.SparseRCNN.DIM_DYNAMIC = 64              # DynamicConv 中间维度
_CC.MODEL.SparseRCNN.HIDDEN_DIM = 256              # proposal feature 维度，须与 FPN.OUT_CHANNELS 一致
_CC.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048        # FFN 中间维度
_CC.MODEL.SparseRCNN.ACTIVATION = "relu"           # FFN 激活函数 relu gelu glu
_CC.MODEL.SparseRCNN.DROPOUT = 0.0                 # 自注意力、动态卷积、FFN共用的dropout
_CC.MODEL.SparseRCNN.NUM_CLS = 1                   # cls 分支 MLP 层数
_CC.MODEL.SparseRCNN.NUM_REG = 3                   # reg 分支 MLP 层数

_CC.MODEL.SparseRCNN.NUM_HEADS = 6                 # DynamicHead 迭代层数（级联）
_CC.MODEL.SparseRCNN.DEEP_SUPERVISION = True       # 中间层是否参与 loss（深度监督）

# loss / matcher（字段名与原版 SparseR-CNN 一致：loss.py / rcnn.py 引用）
_CC.MODEL.SparseRCNN.USE_FOCAL = True              # focal loss（True）或 softmax+CE（False）
_CC.MODEL.SparseRCNN.PRIOR_PROB = 0.01             # focal 模式分类头 bias 初始化
_CC.MODEL.SparseRCNN.ALPHA = 0.25
_CC.MODEL.SparseRCNN.GAMMA = 2.0

_CC.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0            # loss_ce 权重（也是 matcher cost_class）
_CC.MODEL.SparseRCNN.L1_WEIGHT = 5.0               # loss_bbox 权重（也是 matcher cost_bbox）
_CC.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0             # loss_giou 权重（也是 matcher cost_giou）
_CC.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.01       # softmax 模式背景类权重（eos_coef）

# ---- SparseRCNN 冻结开关（few-shot 微调阶段用）----
# SparseRCNN 参数按功能分四组（详见 rcnn.py 的 _apply_freeze）：
#   A proposal queries    init_proposal_boxes / init_proposal_features    → 始终可训练（无开关）             2
#   B 输出头               head_series.{i}.class_logits / bboxes_delta     → 始终可训练（无开关）             24 4*6
#   C cls/reg 适配 MLP     head_series.{i}.cls_module / reg_module         → 始终可训练（无开关）             72 12*6
#   D 特征交互主干          head_series.{i}.self_attn / inst_interact (两个linear三个norm) /
#                         linear1/linear2 / norm1/norm2/norm3             → 由 FREEZE_INTERACTION 控制     144 24*6
# 默认 False（基类预训练：全部可训练）；few-shot 微调时可设 True（冻 D，保 A/B/C 学新类）。
_CC.MODEL.SparseRCNN.FREEZE_INTERACTION = False

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = -1

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = "StandardROIHeads"           # 占位
_CC.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"      # 占位
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
# AdamW 是集合预测的推荐优化器，传统 RCNN 仍可用 SGD。
# BACKBONE_MULTIPLIER 给参数名含 "backbone" 的参数单独缩放学习率 （共包含backbone. head. 两个init_proposal_）
_CC.SOLVER.OPTIMIZER = "ADAMW"           # "SGD" 或 "ADAMW"
_CC.SOLVER.BACKBONE_MULTIPLIER = 1.0     # backbone 学习率倍率（ViT/DINOv3 可设小一些如 0.1）
