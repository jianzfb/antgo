import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.builder import BACKBONES
import torchvision


@BACKBONES.register_module()
class ResnetTV(nn.Module):
    def __init__(self, model, pretrained=True, class_num=None, output=[0,1,2,3]):
        super(ResnetTV, self).__init__()
        self.model = getattr(torchvision.models, model)(pretrained=pretrained)
        if class_num is None:
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
        self.class_num = class_num
        self.output_index = output

    def forward(self, x):
        output = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x4 = self.model.layer1(x)
        x8 = self.model.layer2(x4)
        x16 = self.model.layer3(x8)
        x32 = self.model.layer4(x16)
        output = [x4,x8,x16,x32]
        if self.class_num is None:
            return [output[i] for i in self.output_index]

        return self.model.fc(x32)