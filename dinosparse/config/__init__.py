# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config.instantiate import instantiate
from detectron2.config.lazy import LazyCall, LazyConfig
from detectron2.utils.env import fixup_module_metadata
from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "configurable",
    "instantiate",
    "LazyCall",
    "LazyConfig",
]

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
