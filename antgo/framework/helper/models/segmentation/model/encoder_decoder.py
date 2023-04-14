from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from antgo.framework.helper.models.builder import HEADS, MODELS
from antgo.framework.helper.runner import BaseModule


@MODELS.register_module()
class EncoderDecoder(BaseModule):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck = None,
                 train_cfg = None,
                 test_cfg = None,
                 init_cfg = None):
        super().__init__(init_cfg=init_cfg)
        # if pretrained is not None:
        #     assert backbone.get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.with_neck = False
        if neck is not None:
            self.neck = MODELS.build(neck)
            self.with_neck = True

        self.decode_head = HEADS.build(decode_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, image):
        """Extract features from images."""
        x = self.backbone(image)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, image, segments, **kwargs):
        x = self.extract_feat(image)
        x = self.decode_head(x)

        losses = dict()
        loss_decode = \
            self.decode_head.loss(x, segments.long())
        losses.update(loss_decode)
        return losses

    def forward_test(self, image, **kwargs):
        x = self.extract_feat(image)
        seg_predict = self.decode_head.simple_test(x)
        seg_predict = F.interpolate(seg_predict, size=(image.shape[2:]), mode='bilinear')
        seg_predict = torch.argmax(seg_predict, 1)
        return {
            'pred_segments': seg_predict
        }

    def onnx_export(self, image):
        feat = self.extract_feat(image)
        seg_predict = self.decode_head.simple_test(feat)
        return seg_predict
