# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN
from .dinov3 import DINOv3Backbone, build_dinov3_backbone  # 触发 build_dinov3_backbone 注册

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
