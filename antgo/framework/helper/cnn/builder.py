# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import Registry, build_from_cfg


def build_model_from_cfg(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)


MODELS = Registry('model', build_func=build_model_from_cfg)
