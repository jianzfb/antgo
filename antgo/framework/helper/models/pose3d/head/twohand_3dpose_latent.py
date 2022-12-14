import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from antgo.framework.helper.runner import BaseModule
from antgo.framework.helper.cnn import ConvModule, DepthwiseSeparableConvModule
from antgo.framework.helper.cnn.bricks.waterfall import PWWaterFall, ConvWaterFall
from antgo.framework.helper.models.builder import HEADS
from antgo.framework.helper.models.pose3d.head.layer import make_linear_layers


@HEADS.register_module()
class TwoHand3DPoseLatent(BaseModule):
    def __init__(self, joint_num=21, output_root_hm_shape=24, **kwargs):
        super(TwoHand3DPoseLatent, self).__init__()
        self.joint_num = joint_num  # single hand
        self.output_root_hm_shape = output_root_hm_shape
        # 左手热图
        self.joint_left_enc = nn.Sequential(
            ConvWaterFall(64, 32, 64),
            ConvWaterFall(64, 32, 64)
        )
        self.joint_left_final_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 21, kernel_size=1, bias=True)
        )     

        # 左手latent depth-maps
        self.joint_left_latent_depth = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 21, kernel_size=1, bias=True)         
        )

        # 右手热图+偏移
        self.joint_right_enc = nn.Sequential(
            ConvWaterFall(64, 32, 64),
            ConvWaterFall(64, 32, 64)
        )
        self.joint_right_final_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 21, kernel_size=1, bias=True)
        )
        # 右手latent depth-maps
        self.joint_right_latent_depth = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 21, kernel_size=1, bias=True)            
        )        

        self.root_fc = make_linear_layers([1024,1024,self.output_root_hm_shape],relu_final=False) # 2048->512
        self.hand_fc = make_linear_layers([1024,1024,2],relu_final=False)     

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        if heatmap1d.device.type == 'cpu':
            accu = heatmap1d * torch.arange(self.output_root_hm_shape).float()[None,:]
        else:
            accu = heatmap1d * torch.arange(self.output_root_hm_shape).float().cuda()[None,:]
            
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):
        # left and right hand share
        left_x32, right_x32, x4 = img_feat

        # 左手heatmap + latent
        left_up32 = self.joint_left_enc(left_x32)
        joint_left_h = self.joint_left_final_layer(left_up32)
        joint_left_latent_depth = self.joint_left_latent_depth(left_up32)

        # 右手heatmap + latent
        right_up32 = self.joint_right_enc(right_x32)
        joint_right_h = self.joint_right_final_layer(right_up32)
        joint_right_latent_depth = self.joint_right_latent_depth(right_up32)
        
        # 左右手相对深度
        img_feat = x4
        img_feat_gap = F.avg_pool2d(img_feat, (img_feat.shape[2],img_feat.shape[3])).view(-1, 1024)
        root_heatmap1d = self.root_fc(img_feat_gap)
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1,1)
        
        # 左右手可见性
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))   

        left_depth_and_sigma = None
        right_depth_and_sigma = None
        left_uv_off_and_sigma = None
        right_uv_off_and_sigma = None
        return joint_left_h, joint_left_latent_depth, \
                joint_right_h, joint_right_latent_depth, \
                left_depth_and_sigma, \
                right_depth_and_sigma, \
                left_uv_off_and_sigma, \
                right_uv_off_and_sigma, \
                root_depth, hand_type