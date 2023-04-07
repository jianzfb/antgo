import warnings

import torch.nn as nn
from antgo.framework.helper.cnn import ConvModule, DepthwiseSeparableConvModule
from antgo.framework.helper.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn.bricks.inverted_residual import InvertedResidual
from antgo.framework.helper.cnn.bricks.make_divisible import make_divisible
import torch


@BACKBONES.register_module()
class DDRMobileNetV2(BaseModule):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int], optional): Output from which stages.
            Default: (1, 2, 4, 7).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]

    def __init__(self,
                 in_channels=1,
                 widen_factor=1.,
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None, **kwargs):
        super(DDRMobileNetV2, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(0, 8))):
            raise ValueError('out_indices must be a subset of range'
                             f'(0, 8). But received {out_indices}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []
        self.channels_num = {}
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                in_channels= self.in_channels * 2 if i in [5] else None
            )
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
            self.channels_num[i] = out_channels
        
        # 编码通道数
        self.encoder_ch = 64
        
        # 1,    2,      4,      6
        # 1/4,  1/8,    1/16,   1/32
        self.down4_to_16 = \
                ConvModule(
                    self.encoder_ch, 
                    self.channels_num[4],
                    kernel_size=3,
                    stride=4, 
                    padding=(int)((4*(3-1)+1-4)//2+1), 
                    dilation=4, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    self.encoder_ch, 
                    self.channels_num[6],
                    kernel_size=3,
                    stride=8, 
                    padding=(int)((6*(3-1)+1-8)//2+1), 
                    dilation=6, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))             

        self.compression8_to_4 = \
                ConvModule(
                    self.channels_num[2], 
                    self.encoder_ch, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    self.channels_num[4], 
                    self.encoder_ch, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        
        self.layer4_1 = DepthwiseSeparableConvModule(
            in_channels=self.channels_num[1],
            out_channels=self.encoder_ch,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_2 = DepthwiseSeparableConvModule(
            in_channels=self.encoder_ch,
            out_channels=self.encoder_ch,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_3 = DepthwiseSeparableConvModule(
            in_channels=self.encoder_ch,
            out_channels=self.encoder_ch,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_4 = DepthwiseSeparableConvModule(
            in_channels=self.encoder_ch,
            out_channels=self.encoder_ch,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_5 = DepthwiseSeparableConvModule(
            in_channels=self.encoder_ch+256,
            out_channels=self.encoder_ch,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )    
      
        # 1/4
        # self.encoder_enchance = WaterFall(self.encoder_ch, self.encoder_ch, self.encoder_ch)

        # 1/32
        self.enchance = \
            ConvModule(
                    self.channels_num[6]*2, 
                    256,
                    kernel_size=3,
                    stride=1, 
                    padding=1, 
                    act_cfg=None,
                    norm_cfg=dict(type='BN'))

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio, in_channels=None):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    in_channels if (in_channels is not None and i == 0) else self.in_channels,
                    out_channels,
                    mid_channels=int(round(self.in_channels * expand_ratio)),
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        out_layer_1 = None
        out_layer_2 = None
        out_layer_3 = None
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                if i == 1:
                    # 1/4
                    out_layer_1 = self.layer4_1(x)
                elif i == 2:
                    # 1/8
                    # merge from 1/4
                    x_ = self.layer4_2(out_layer_1)       # channel 32
                    out_layer_2 = x_ + F.interpolate(
                                    self.compression8_to_4(x),
                                    scale_factor=2,
                                    mode='bilinear')
                    out_layer_2 = F.relu(out_layer_2)       # 1/4
                elif i == 4:
                    # 1/16
                    # merge from 1/4
                    # down_x = F.relu(x + self.down4_to_16(out_layer_2))
                    down_x = F.relu(torch.cat([x, self.down4_to_16(out_layer_2)], dim=1))
                    x_ = self.layer4_3(out_layer_2)
                    out_layer_3 = x_ + F.interpolate(
                                    self.compression16_to_4(x),
                                    scale_factor=4,
                                    mode='bilinear')
                    out_layer_3 = F.relu(out_layer_3)
                    x = down_x   
                elif i == 6:
                    # down_x = F.relu(x + self.down4_to_32(out_layer_3))
                    down_x = F.relu(torch.cat([x, self.down4_to_32(out_layer_3)], dim=1))
                    x = down_x

        # 在1/32分辨率，使用瀑布结果增强全局编码
        # 1/32 分辨率 4x4
        x = self.enchance(x)

        out_layer_4 = self.layer4_4(out_layer_3)
        out_layer = self.layer4_5(
            F.relu(torch.cat([out_layer_4, F.interpolate(x, scale_factor=8, mode='bilinear')], dim=1))
        )

        # 在1/4分辨率，使用瀑布结果抓取多视野信息
        # out_layer = self.encoder_enchance(out_layer)
        return tuple([out_layer, x])

    def init_weights(self):
        # pretrained = '/opt/tiger/handtt/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
        # # dd = self.load_state_dict(pretrained)
        # dd = torch.load(pretrained, map_location='cpu')['state_dict']

        # filter_state_dict = {}
        # for k, v in dd.items():
        #     filter_k = '.'.join(k.split('.')[1:])
        #     filter_state_dict[filter_k] = v

        # self.load_state_dict(filter_state_dict, strict=False)
        pass
