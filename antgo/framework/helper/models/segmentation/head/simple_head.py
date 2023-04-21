import torch
import torch.nn as nn
from antgo.framework.helper.cnn.bricks import *
from antgo.framework.helper.models.builder import HEADS, MODELS, build_loss
from antgo.framework.helper.runner import BaseModule
import torch.nn.functional as F


@HEADS.register_module()
class SimpleHead(BaseModule):
    def __init__(self, 
                 in_channels=160, 
                 channels=32, 
                 out_channels=2, 
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 align_corners=False,
                 ignore_index=None,
                 **kwargs):
        super().__init__(init_cfg=kwargs.get('init_cfg', None))
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels

        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.align_corners = align_corners

    def cls_seg(self, feat):
        """Classify each pixel."""
        # if self.dropout is not None:
        #     feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = inputs[-1]

        feats = self.bottleneck(inputs)
        output = self.cls_seg(feats)
        return output

    def loss(self, seg_logits, seg_labels):
        loss_dict = dict()
        seg_logits = F.interpolate(seg_logits, seg_labels.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_labels = seg_labels.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss_dict['loss_seg'] = loss_decode(
                seg_logits,
                seg_labels,
                ignore_index=self.ignore_index)

        return loss_dict
    
    def simple_test(self, x):
        x = self.forward(x)
        x = torch.softmax(x, 1)
        return x