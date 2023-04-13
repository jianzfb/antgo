import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
from antgo.framework.helper.cnn import ConvModule,DepthwiseSeparableConvModule


class InverseBottleneck(nn.Module):
    expansion = 1
    exp = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(InverseBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("InverseBottleneck only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in InverseBottleneck")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes * self.exp, 1, 1, 0)
        self.bn1 = norm_layer(planes * self.exp)
        self.relu1 = nn.ReLU(inplace=True)        
        self.conv2 = nn.Conv2d(
                    planes * self.exp,
                    planes * self.exp,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=planes * self.exp,
                    bias=False,
                    dilation=1,
                )
        self.bn2 = norm_layer(planes * self.exp)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes * self.exp, planes, 1, 1, 0)
        self.bn3 = norm_layer(planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


@BACKBONES.register_module()
class DDRKetNetF(nn.Module):
    def __init__(self, architecture='resnet34', in_channels=1):
        super(DDRKetNetF, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        assert architecture in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        layers = {
            "resnet34": [1, 1, 1, 2, 2, 1, 4, 4],
        }
        self.inplanes = 32
        if architecture == "resnet18" or architecture == "resnet34":
            self.block = InverseBottleneck
        else:
            self.block = InverseBottleneck
        self.layers = layers[architecture]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = self.norm_layer(32, eps=1e-5, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.block.exp = 3
        self.layer1 = self.make_layer(self.block, 32, self.layers[0], stride=1)
        self.layer2 = self.make_layer(self.block, 32, self.layers[1], stride=2)
        self.block.exp = 6
        self.layer3 = self.make_layer(self.block, 32, self.layers[2], stride=1)
        self.layer4 = self.make_layer(self.block, 64, self.layers[3], stride=2)
        self.block.exp = 4
        self.layer5 = self.make_layer(self.block, 96, self.layers[4], stride=1)
        self.layer6 = self.make_layer(self.block, 128, self.layers[5], stride=2)
        self.block.exp = 6
        self.layer7 = self.make_layer(self.block, 128, self.layers[6], stride=1)
        self.layer8 = self.make_layer(self.block, 160, self.layers[7], stride=2)

        # 1/4 ,  1/8, 1/16, 1/32
        # (dilation(k-1)+1-stride)/2
        self.down4_to_16 = \
                ConvModule(
                    32, 
                    128,
                    kernel_size=3,
                    stride=4, 
                    padding=(int)((4*(3-1)+1-4)//2+1), 
                    dilation=4, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.down4_to_32 = \
                ConvModule(
                    32, 
                    160,
                    kernel_size=3,
                    stride=8, 
                    padding=(int)((6*(3-1)+1-8)//2+1), 
                    dilation=6, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))             

        self.compression8_to_4 = \
                ConvModule(
                    96, 
                    32, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
        self.compression16_to_4 = \
                ConvModule(
                    128, 
                    32, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))

        self.compression32_to_4 = \
                ConvModule(
                    160, 
                    32, 
                    kernel_size=1, 
                    act_cfg=None, 
                    norm_cfg=dict(type='BN'))
    
        self.layer4_1 = nn.Sequential(
            ConvModule(
                32,
                32,
                kernel_size=3,
                padding=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN'),
            )
        )
        self.layer4_2 = nn.Sequential(
            ConvModule(
                32,
                32,
                kernel_size=3,
                padding=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN'),
            )
        )      
        self.layer4_3 = nn.Sequential(
            ConvModule(
                32,
                32,
                kernel_size=3,
                padding=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN'),
            )
        )      
        self.layer4_4 = nn.Sequential(
            ConvModule(
                32,
                32,
                kernel_size=3,
                padding=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN'),
            )
        )       

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 32 * h/2 * w/2
        x = self.layer1(x)
        x32 = self.layer2(x)    # 32 * h/4 * w/4
        x32 = self.layer3(x32)  # 64 * h/4 * w/4

        out_layer_1 = self.layer4_1(x32)

        x16 = self.layer4(x32)  # 64 * h/8 * w/8
        x16 = self.layer5(x16)  # 96 * h/8 * w/8

        # 
        out_layer_2 = \
            self.layer4_2(out_layer_1) + F.interpolate(
                                    self.compression8_to_4(x16),
                                    scale_factor=2,
                                    mode='bilinear')
        
        x8 = self.layer6(x16)   # 128 * h/16 * w/16
        x8 = self.layer7(x8)    # 128 * h/16 * w/16

        down_x = F.relu(x8 + self.down4_to_16(out_layer_2))
        out_layer_3 = \
            self.layer4_3(out_layer_2) + F.interpolate(
                                self.compression16_to_4(x8),
                                scale_factor=4,
                                mode='bilinear')
        x8 = down_x
        x4 = self.layer8(x8)    # 160 * h/32 * w/32
        x4 = F.relu(x4 + self.down4_to_32(out_layer_3))

        out_layer = \
            self.layer4_4(out_layer_3) + F.interpolate(
                self.compression32_to_4(x4), 
                scale_factor=8, 
                mode='bilinear')
        return [out_layer]

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)
