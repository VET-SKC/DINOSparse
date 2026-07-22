# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec

from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    build_backbone,
)
from .meta_arch import (
    META_ARCH_REGISTRY,
    SparseRCNN,
    build_model,
)
from .heads import (
    DynamicHead,
    SetCriterion,
    HungarianMatcher,
)
from .postprocessing import detector_postprocess


_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
