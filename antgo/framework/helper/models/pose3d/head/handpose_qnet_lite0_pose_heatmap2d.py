""" copied from handpose/models/simplepose_QNetLite0.py
"""
# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn
from antgo.framework.helper.models.builder import HEADS


@HEADS.register_module()
class SimpleQNetLite0PoseHeatmap2D(nn.Module):
    def __init__(self, num_deconv_filters, num_joints):
        super(SimpleQNetLite0PoseHeatmap2D, self).__init__()
        self.deconv_dim = num_deconv_filters
        self.deconv_layer0, self.deconv_layer1 = self._make_deconv_layer()

        self.fuse_keypoint = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

        self.fuse_up16 = nn.Sequential(
            nn.Conv2d(self.deconv_dim[1], 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Conv2d(
            128, num_joints, kernel_size=3, stride=1, padding=1
        )  # up32 = torch.cat[up32, x32] x32's channel is 32
        self.final_offset_layer = nn.Conv2d(
            128, num_joints * 2, kernel_size=3, stride=1, padding=1
        )  # up32 = torch.cat[up32, x32]  x32's channel is 32

    def _make_deconv_layer(self):
        deconv_layer0 = []
        deconv_layer1 = []
        # deconv_layer2 = []
        # deconv1 = nn.ConvTranspose2d(
        #     2048, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        deconv1 = nn.Upsample(scale_factor=2, mode="nearest")  # ,align_corners=False
        # deconv1 = nn.Upsample(scale_factor=2, mode='nearest')
        conv1_1x1 = nn.Conv2d(
            160, self.deconv_dim[0], kernel_size=3, stride=1, padding=1, bias=False
        )  # 2048-->512 for resnet18 @lyp
        bn1 = nn.BatchNorm2d(self.deconv_dim[0])
        # deconv2 = nn.ConvTranspose2d(
        #     self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        deconv2 = nn.Upsample(scale_factor=2, mode="nearest")
        # deconv2 = nn.Upsample(scale_factor=2, mode='nearest')
        conv2_1x1 = nn.Conv2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(self.deconv_dim[1])

        deconv_layer0.append(deconv1)
        deconv_layer0.append(conv1_1x1)
        deconv_layer0.append(bn1)
        deconv_layer0.append(nn.ReLU(inplace=True))

        deconv_layer1.append(deconv2)
        deconv_layer1.append(conv2_1x1)
        deconv_layer1.append(bn2)
        deconv_layer1.append(nn.ReLU(inplace=True))

        # deconv_layer2.append(deconv3)
        # deconv_layer2.append(conv3_1x1)
        # deconv_layer2.append(bn3)
        # deconv_layer2.append(nn.ReLU(inplace=True))

        return [nn.Sequential(*deconv_layer0), nn.Sequential(*deconv_layer1)]

    def forward(self, *args):
        # x32, x16, x8, x4 = self.preact(im)  # c: 64, 128, 192, 256

        x32, x16, x8, x4 = args[0]
        # his_feat = self.keypoint_layer(his_feat)
        # x4_inv = FTL_invK(x4, K_inv)
        x4_inv = x4
        # x4_fuse = torch.cat([x4, his_feat], dim=1)

        x4_fuse = self.fuse_keypoint(x4_inv)

        up8 = self.deconv_layer0(x4_fuse)

        up16 = self.deconv_layer1(up8)

        up16 = self.fuse_up16(up16)

        uv_heatmap = self.final_layer(up16)
        uv_off = self.final_offset_layer(up16)
        return [uv_heatmap, uv_off]

