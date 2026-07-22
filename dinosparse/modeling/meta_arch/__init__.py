# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import SparseRCNN  # 触发 SparseRCNN 注册到 META_ARCH_REGISTRY


__all__ = list(globals().keys())
