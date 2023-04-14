import torch
import torch.nn as nn
import torch.nn.functional as F
from antgo.framework.helper.models.pose3d.backbone import *
from antgo.framework.helper.cnn.backbone import *
from antgo.framework.helper.models.pose3d.head import *

from antgo.framework.helper.models.pose3d.losses.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss
from antgo.framework.helper.models.pose3d.losses.gaussian_focal_loss import GaussianFocalLoss
from antgo.framework.helper.runner import *
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head
import math


@MODELS.register_module()
class DDRTwoHandPose3DModel(BaseModule):
    def __init__(self, backbone, pose_head, **kwargs):
        super(DDRTwoHandPose3DModel, self).__init__()

        # modules
        self.backbone_net = build_backbone(backbone)
        self.pose_net = build_head(pose_head)
        # loss functions
        self.hand_type_loss = HandTypeLoss()

        # 左右手heatmap和offset
        # self.left_joint_heatmap_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.left_joint_offset_loss = torch.nn.MSELoss(reduction='none')
        # self.right_joint_heatmap_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.right_joint_offset_loss = torch.nn.MSELoss(reduction='none')
        self.heatmap_loss = GaussianFocalLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()

        # 左右手深度和offset
        self.left_joint_depth_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.left_joint_doffset_loss = torch.nn.MSELoss(reduction='none')
        self.right_joint_depth_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.right_joint_doffset_loss =  torch.nn.MSELoss(reduction='none')

    def forward(self, image, bbox, camera, **kwargs):
        input_img = image
        img_feat = self.backbone_net(input_img)

        joint_left_h, \
        joint_left_offset, \
        joint_right_h, \
        joint_right_offset, \
        left_depth, \
        left_d_off, \
        right_depth, \
        right_d_offset, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        
        return torch.zeros((1))
        
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        # nn.init.constant_(m.bias,0)        
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        # nn.init.constant_(m.bias,0)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)        
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01) 
        # nn.init.constant_(m.bias,0)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)               



# def get_ddr_model():
#     # backbone_net = BackboneNet(backbone_type=backbone_type)
#     # pose_net = SimplePoseNet(joint_num)
#     # backbone_net = DDRMobileNetV2(in_channels=3)
#     backbone_net = DDRResNetBackbone(resnet_type=34)
#     # backbone_net.init_weights()
    
#     # pose_net = DDRSimplePoseNet(21)
#     pose_net = DDRSimplePoseNetBaseline(21)
#     model = DDRModel(backbone_net, pose_net)
#     return model

