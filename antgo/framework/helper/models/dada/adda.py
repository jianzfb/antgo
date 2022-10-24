"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse

from sys import maxsize
from numpy.lib.function_base import diff
import torch
from torch._C import device
from antgo.framework.helper.models.builder import DETECTORS, build_model
from antgo.framework.helper.base_module import BaseModule
import numpy as np
import cv2
import os
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from collections.abc import Sequence


class Adda(BaseModule):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(Adda, self).__init__(
            dict(
                source_model=build_model(model),
                target_model=build_model(model)
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

        # 构建discriminator
        pass

    def forward_train(self, image, image_metas, **kwargs):
        super().forward_train(image, image_metas, **kwargs)

        source_feature = self.source_model.extract_feature
        target_feature = self.target_model.extract_feature
    
        # 
        # 

        loss = {}
        return loss

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msg):
        pass