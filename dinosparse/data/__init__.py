# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.data import samplers
from detectron2.data import transforms  # isort:skip
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.data.common import DatasetFromList, MapDataset, ToIterableDataset
from .build import (
    build_batch_data_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    load_proposals_into_dataset,
    print_instances_class_histogram,
)
from .dataset_mapper import DatasetMapper
from . import datasets  # isort:skip

# ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if not k.startswith("_")]
