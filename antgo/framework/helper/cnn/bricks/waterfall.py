import torch
import torch.nn as nn
from antgo.framework.helper.cnn import ConvModule, DepthwiseSeparableConvModule


class PWWaterFall(nn.Module):
    def __init__(self, in_channels, middle_channels=32, out_channels=32):
        super().__init__()
        self.waterfall_block_root = ConvModule(
                (int)(in_channels),
                middle_channels,
                1,
                padding=0,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
        )                

        self.waterfall_block_0 = DepthwiseSeparableConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=(int)((2*(3-1)+1-1)//2),
            dilation=2,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')     
        )
        self.waterfall_block_1 = DepthwiseSeparableConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=(int)((4*(3-1)+1-1)//2),
            dilation=4,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.waterfall_block_2 = DepthwiseSeparableConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=(int)((6*(3-1)+1-1)//2),
            dilation=6,
            dw_act_cfg=dict(type='ReLU'), dw_norm_cfg=dict(type='BN'), pw_norm_cfg=dict(type='BN'),pw_act_cfg=dict(type='ReLU')   
        )
        self.waterfall_block_fusion = ConvModule(
                (int)(middle_channels * 4),
                out_channels,
                1,
                padding=0,
                norm_cfg=dict(type='BN')
        )        

    def forward(self, x):
        x_root = self.waterfall_block_root(x)
        x0 = self.waterfall_block_0(x_root)
        x1 = self.waterfall_block_1(x0)
        x2 = self.waterfall_block_2(x1)
        x3 = torch.cat([x_root,x0,x1,x2], dim=1)
        x = self.waterfall_block_fusion(x3)
        return x

class ConvWaterFall(nn.Module):
    def __init__(self, in_channels, middle_channels=32, out_channels=32, out_relu=False):
        super().__init__()
        self.block_root = ConvModule(
            (int)(in_channels),
            middle_channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )                

        self.block_0 = ConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        self.block_1 = ConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.block_2 = ConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        if out_relu:
            self.block_fuse = ConvModule(
                in_channels=(middle_channels*4),
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )     
        else:
            self.block_fuse = ConvModule(
                in_channels=(middle_channels*4),
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                norm_cfg=dict(type='BN'),
                act_cfg=None
            )     
    
    def forward(self, x):
        x_root = self.block_root(x)
        x0 = self.block_0(x_root)
        x1 = self.block_1(x0)
        x2 = self.block_2(x1)
        x3 = torch.cat([x_root, x0, x1, x2], dim=1)
        x = self.block_fuse(x3)
        return x

class ConvWaterFallV2(nn.Module):
    def __init__(self, in_channels, middle_channels=32, out_channels=32):
        super().__init__()
        self.block_root = ConvModule(
            (int)(in_channels),
            middle_channels,
            1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )                

        self.block_0 = ConvModule(
            in_channels=(int)(in_channels),
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        self.block_1 = ConvModule(
            in_channels=(int)(in_channels),
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.block_2 = ConvModule(
            in_channels=(int)(in_channels),
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        self.block_fuse = ConvModule(
            in_channels=(middle_channels*4),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )     

    def forward(self, x):
        x_root = self.block_root(x)
        x0 = self.block_0(x)
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = torch.cat([x_root, x0, x1, x2], dim=1)
        x = self.block_fuse(x3)
        return x

