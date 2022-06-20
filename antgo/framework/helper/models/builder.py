# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from antgo.framework.helper.cnn import MODELS as CV_MODELS
from antgo.framework.helper.utils import Registry

MODELS = Registry('models', parent=CV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_model(cfg):
    """Build detector."""
    return MODELS.build(
        cfg, 
        default_args=dict(
            train_cfg=None,
            test_cfg=None
        )
    )
