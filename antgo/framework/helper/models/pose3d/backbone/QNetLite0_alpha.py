import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        dcn=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


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
        norm_layer=None,
        dcn=None,
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
        self.conv2 = conv3x3(planes * self.exp, planes * self.exp, stride, planes * self.exp)
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


class Bottleneck(nn.Module):
    # expansion = 4
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get("FALLBACK_ON_STRIDE", False)
            self.with_modulated_dcn = dcn.get("MODULATED", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.deformable_groups = dcn.get("DEFORM_GROUP", 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27

            self.conv2_offset = nn.Conv2d(
                planes, self.deformable_groups * offset_channels, kernel_size=3, stride=stride, padding=1
            )
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                deformable_groups=self.deformable_groups,
                bias=False,
            )

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)  # 4->2
        self.bn3 = norm_layer(planes * 2, momentum=0.1)  # 4->2
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, : 18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups :, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


@BACKBONES.register_module()
class KetNetF(nn.Module):
    """KetNetF"""

    def __init__(self, architecture, in_channels=1, out_indices=[1,2,3,4]):
        super(KetNetF, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.stage_with_dcn=(False, False, False, False, False, False, False, False)
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
        stage_dcn = [None for with_dcn in self.stage_with_dcn]

        self.out_indices = sorted(out_indices)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = self.norm_layer(32, eps=1e-5, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.block.exp = 3
        self.layer1 = self.make_layer(self.block, 32, self.layers[0], stride=1, dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 32, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.block.exp = 6
        self.layer3 = self.make_layer(self.block, 32, self.layers[2], stride=1, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 64, self.layers[3], stride=2, dcn=stage_dcn[3])
        self.block.exp = 4
        self.layer5 = self.make_layer(self.block, 96, self.layers[4], stride=1, dcn=stage_dcn[4])
        self.layer6 = self.make_layer(self.block, 128, self.layers[5], stride=2, dcn=stage_dcn[5])
        self.block.exp = 6
        self.layer7 = self.make_layer(self.block, 128, self.layers[6], stride=1, dcn=stage_dcn[6])
        self.layer8 = self.make_layer(self.block, 160, self.layers[7], stride=2, dcn=stage_dcn[7])
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 32 * h/4 * w/4
        x = self.layer1(x)
        outs = []
        if 0 in self.out_indices:
            outs.append(x)
        x32 = self.layer2(x)  # 32 * h/4 * w/4
        x32 = self.layer3(x32)  # 64 * h/4 * w/4
        if 1 in self.out_indices:
            outs.append(x32)
        x16 = self.layer4(x32)  # 64 * h/8 * w/8
        x16 = self.layer5(x16)  # 96 * h/8 * w/8
        if 2 in self.out_indices:
            outs.append(x16)
        
        x8 = self.layer6(x16)  # 128 * h/16 * w/16
        x8 = self.layer7(x8)  # 128 * h/16 * w/16
        if 3 in self.out_indices:
            outs.append(x8)

        x4 = self.layer8(x8)  # 160 * h/32 * w/32
        if 4 in self.out_indices:
            outs.append(x4)
        return outs

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, dcn=dcn))

        return nn.Sequential(*layers)
