import warnings

import torch
import torch.nn.functional as F
from torch import nn

from antgo.framework.helper.models.classification.losses.accuracy import Accuracy
from antgo.framework.helper.models.classification.losses.cross_entropy_loss import *
from ...builder import *


@HEADS.register_module()
class ClsHead(nn.Module):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=None),
                 init_cfg=dict(use_softmax=True)):
        super().__init__()

        assert isinstance(loss, dict)
        self.compute_loss = CrossEntropyLoss(loss_weight=loss['loss_weight'], class_weight=loss['class_weight'])
        self.use_softmax = init_cfg.get('use_softmax', True)

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
            
        losses = self.loss(cls_score, gt_label.view(-1), **kwargs)
        return losses

    def simple_test(self, x, **kwargs):
        """Inference without augmentation.

    Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        cls_score = self.pre_logits(x)

        if self.use_softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score
        return pred

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x
