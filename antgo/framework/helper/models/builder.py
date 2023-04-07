import warnings

from antgo.framework.helper.cnn import MODELS as CV_MODELS
from antgo.framework.helper.utils import Registry

MODELS = Registry('models', parent=CV_MODELS)

CLASSIFIERS = MODELS
BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
DISTILLER = MODELS
DISTILL_LOSSES = MODELS


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


def build_distiller(cfg, teacher_cfg, student_cfg):
    """Build distiller."""
    # if train_cfg is not None or test_cfg is not None:
    #     warnings.warn(
    #         'train_cfg and test_cfg is deprecated, '
    #         'please specify them in model', UserWarning)
    # assert cfg.get('train_cfg') is None or train_cfg is None, \
    #     'train_cfg specified in both outer field and model field '
    # assert cfg.get('test_cfg') is None or test_cfg is None, \
    #     'test_cfg specified in both outer field and model field '
    """Build detector."""
    return MODELS.build(
        cfg, 
        default_args=dict(
            teacher_cfg=teacher_cfg,
            student_cfg=student_cfg
        )
    )    

def build_distill_loss(cfg):
    """Build distill loss."""
    return DISTILL_LOSSES.build(cfg)