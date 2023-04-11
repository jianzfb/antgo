from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck
from antgo.framework.helper.runner import BaseModule


@MODELS.register_module()
class KeypointNet(BaseModule):
    def __init__(
        self, backbone, head, neck=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.with_neck = False
        if neck is not None:
            self.neck = build_neck(neck)
            self.with_neck = True
        self.head = build_head(head)

    def _compute_loss_with_heatmap(self, uv_heatmap, uv_off, label, meta) -> Dict[str, torch.Tensor]:
        """compute loss"""
        batch_size = uv_heatmap.shape[0]
        joints_vis_2d = meta["joints_vis_2d"]  # bs x 21
        joints_vis_2d = (joints_vis_2d.sum(dim=1) > 0).long()

        joint_mask = joints_vis_2d.reshape(batch_size, 1, 1, 1).repeat(
            1, uv_heatmap.shape[1], uv_heatmap.shape[2], uv_heatmap.shape[3]
        )

        # 2d heatmap + depth loass
        loss_hm, loss_offx, loss_offy, loss_hmz, loss_offz = self.heatmap_loss(
            (uv_heatmap, uv_off), label, joint_mask, joints_vis_2d, meta
        )

        outloss = dict()
        outloss["loss_uv_hm"] = loss_hm
        outloss["loss_xy_offset"] = loss_offx + loss_offy

        loss = (
            loss_hm + 0.1 * (loss_offx + loss_offy)
        )

        outloss["loss"] = loss
        return outloss

    def forward_train(self, image, **kwargs):
        output = self.backbone(image)   # x32,x16,x8,x4
        output = self.head(output)      # uv_heatmap, uv_off

        label = kwargs.get('label')
        uv_heatmap, uv_off = output
        loss_output = self._compute_loss_with_joint_loss_with_heatmap(uv_heatmap, uv_off, label)
        return loss_output

    def forward_test(self, image, **kwargs):
        pass 
