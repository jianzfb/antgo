from ...builder import CLASSIFIERS, build_backbone, build_head, build_neck

import torch
from .base import *
from antgo.framework.helper.runner import BaseModule


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None, **kwargs):
        super(BaseClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

    def extract_feat(self, img, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def forward_train(self, image, label=None, **kwargs):
        """Forward computation during training.
        Args:
            image (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                should be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(image)

        if label is None:
            return x[0]
        losses = dict()
        loss = self.head.forward_train(x, label)

        losses.update(loss)
        return losses

    def loss(self, x, label, **kwargs):
        losses = dict()
        loss = self.head.forward_train(x, label)

        losses.update(loss)
        return losses

    def simple_test(self, image, image_meta=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(image)
        res = self.head.simple_test(x, **kwargs)
        res = {
            'pred': res
        }
        return res

    def onnx_export(self, image):
        x = self.extract_feat(image)
        res = self.head.simple_test(x)
        return res
