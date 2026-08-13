# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import SparseRCNN  # 触发 SparseRCNN 注册到 META_ARCH_REGISTRY
from .depth import MonoDepthMetaArch  # 单目深度估计
from .seg import SemSegMetaArch  # 语义分割


__all__ = list(globals().keys())
