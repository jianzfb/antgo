import torch.nn as nn
import torch
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn import ConvModule,DepthwiseSeparableConvModule

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1)//2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.hardswish = nn.Hardswish()
        self.hardswish = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out

class DepthSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_kernel_size, use_se=False, pw_kernel_size=1):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = StemConv(
            in_channels, in_channels, kernel_size=dw_kernel_size,
            stride=stride, num_groups=in_channels)
        if self.use_se:
            self.se = SEModule(in_channels)
        self.pw_conv = StemConv(in_channels, out_channels, kernel_size=pw_kernel_size, stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)

        return x

class DPM(nn.Module):
    def __init__(self, in_channels, middle_channels=32, out_channels=32):
        super().__init__()
        self.multi_block_0 = ConvModule(
                    in_channels, 
                    middle_channels,
                    kernel_size=1,
                    padding=0, 
                    act_cfg=dict(type='ReLU'), 
                    norm_cfg=dict(type='BN')
        )        
        self.multi_block_1 = DepthwiseSeparableConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=(int)((2*(3-1)+1-1)//2),
            dilation=2,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.multi_block_2 = DepthwiseSeparableConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=(int)((4*(3-1)+1-1)//2),
            dilation=4,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.multi_block_fusion = ConvModule(
                middle_channels*3,
                out_channels,
                1,
                padding=0,
                act_cfg=dict(type='ReLU'), 
                norm_cfg=dict(type='BN'),
        )        
    
    def forward(self, x):
        x0 = self.multi_block_0(x)
        x1 = self.multi_block_1(x0)
        x2 = self.multi_block_2(x1)
        x3 = torch.cat([x0,x1,x2], dim=1)
        x = self.multi_block_fusion(x3)
        return x


@BACKBONES.register_module()
class DDRLCNet(nn.Module):
    def __init__(self, cfgs, in_channels=1, block=DepthSepConvBlock, num_classes=1000, dropout=0.2, scale=1.0, class_expand=1280):
        super(DDRLCNet, self).__init__()
        self.cfgs = cfgs
        self.class_expand = class_expand
        self.block = block

        self.conv1 = StemConv(
            in_channels=in_channels,
            kernel_size=3,
            out_channels=make_divisible(16 * scale),
            stride=2)
        self.stages = torch.nn.ModuleList()
        self.channels_num = {}
        for cfg_i, cfg in enumerate(self.cfgs):
            layers = []
            out_channel = 0
            for k, inplanes, planes, stride, use_se in cfg:
                in_channel = make_divisible(inplanes * scale)
                out_channel = make_divisible(planes * scale)
                layers.append(block(in_channel, out_channel, stride=stride, dw_kernel_size=k, use_se=use_se))

            self.channels_num[cfg_i] = out_channel
            self.stages.append(nn.Sequential(*layers))

        # 2   ,  4,   10,   12
        # 1/4 ,  1/8, 1/16, 1/32
        # (dilation(k-1)+1-stride)/2
        self.down4_to_16 = \
                ConvModule(
                    32, 
                    self.channels_num[10],
                    kernel_size=3,
                    stride=4, 
                    padding=(int)((4*(3-1)+1-4)//2+1), 
                    dilation=4, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    32, 
                    self.channels_num[12],
                    kernel_size=3,
                    stride=8, 
                    padding=(int)((6*(3-1)+1-8)//2+1), 
                    dilation=6, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))             

        self.compression8_to_4 = \
                ConvModule(
                    self.channels_num[4], 
                    32, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    self.channels_num[10], 
                    32, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))

        self.layer4_1 = DepthwiseSeparableConvModule(
            in_channels=self.channels_num[2],
            out_channels=32,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_2 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )
        self.layer4_3 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            dilation=1,
            act_cfg=None, dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN')
        )

        # (dilation(k-1)+1-stride)/2
        self.multi_block_0 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(int)((2*(3-1)+1-1)//2),
            dilation=2,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')     
        )
        self.multi_block_1 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(int)((4*(3-1)+1-1)//2),
            dilation=4,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.multi_block_2 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(int)((6*(3-1)+1-1)//2),
            dilation=6,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.multi_block_fusion = ConvModule(
                128,
                32,
                1,
                padding=0,
                norm_cfg=dict(type='BN'),
        )        

        # DPM
        self.dpm = DPM(self.channels_num[12], 32, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        out_layer = None # 1/4 resolution
        for stage_i, stage_module in enumerate(self.stages):
            x = stage_module(x)
            if stage_i == 2:
                out_layer = self.layer4_1(x)        # channel 32
            elif stage_i == 4:
                # 1/8
                # merge from 1/4
                x_ = self.layer4_2(out_layer)       # channel 32
                out_layer = x_ + F.interpolate(
                                self.compression8_to_4(x),
                                scale_factor=2,
                                mode='bilinear')
                out_layer = F.relu(out_layer)       # 1/4
            elif stage_i == 10:
                # 1/16
                # merge from 1/4
                down_x = F.relu(x + self.down4_to_16(out_layer))
                x_ = self.layer4_3(out_layer)
                out_layer = x_ + F.interpolate(
                                self.compression16_to_4(x),
                                scale_factor=4,
                                mode='bilinear')
                out_layer = F.relu(out_layer)
                x = down_x     
            elif stage_i == 12:
                # 1/32
                down_x = F.relu(x + self.down4_to_32(out_layer))
                # x_ = self.layer4_4(out_layer)
                # out_layer = x_ + F.interpolate(
                #                 self.compression32_to_4(x),
                #                 scale_factor=8,
                #                 mode='bilinear')    
                x = down_x

        # fuse 1/4 and 1/32
        out_layer = out_layer + F.interpolate(self.dpm(x), scale_factor=8, mode='bilinear') 

        # 瀑布式结构（多dilation操作实现视野捕捉）
        branch_1 = self.multi_block_0(out_layer)
        branch_2 = self.multi_block_1(branch_1)
        branch_3 = self.multi_block_2(branch_2)
        branch_all = torch.cat([out_layer, branch_1, branch_2, branch_3], dim=1)
        out_layer = self.multi_block_fusion(branch_all)
        return out_layer,
    
    def init_weights(self):
        pass