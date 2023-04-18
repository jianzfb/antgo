import torch
import torch.nn as nn
from antgo.framework.helper.cnn.bricks import *
from antgo.framework.helper.models.builder import HEADS, MODELS, build_loss
from antgo.framework.helper.runner import BaseModule
import torch.nn.functional as F


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.
    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHead(BaseModule):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.
    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, 
                 dilations=(1, 6, 12, 18), 
                 in_channels=160, 
                 channels=32, 
                 out_channels=2, 
                 dropout_ratio=0.1,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 align_corners=False,
                 ignore_index=None,
                 **kwargs):
        super().__init__(init_cfg=kwargs.get('init_cfg', None))
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = inputs[-1]
        aspp_outs = [
            F.interpolate(self.image_pool(x), x.size()[2:], None, mode='bilinear', align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def loss_by_feat(self, seg_logits, seg_labels):
        """Compute segmentation loss.
        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        loss = dict()
        seg_logits = F.interpolate(seg_logits, seg_labels.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_labels = seg_labels.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss['loss_seg'] = loss_decode(
                seg_logits,
                seg_labels,
                ignore_index=self.ignore_index)

        return loss
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
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