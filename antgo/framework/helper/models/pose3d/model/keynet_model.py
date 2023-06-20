from typing import Dict

import torch
from torch import nn
from antgo.framework.helper.models.builder import MODELS, build_backbone, build_head, build_neck
from antgo.framework.helper.runner import BaseModule
import numpy as np
import torch.nn.functional as F


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
        
        self.ohem = True
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none' if self.ohem else 'mean')
        self.criterion = torch.nn.MSELoss(reduction='none' if self.ohem else 'mean')
        self.offset_loss_weight = 0.1
        self.heatmap_loss_weight = 1.0
        if train_cfg is not None:
            self.offset_loss_weight = train_cfg.get('offset_loss_weight', 0.1)
            self.heatmap_loss_weight = train_cfg.get('heatmap_loss_weight', 1.0)
    
    def _loss(self, pred_heatmap, pred_offset_xy, gt_heatmap, joint_mask, heatmap_mask, offset_x, offset_y):
        hard_weight = 2
        mid_weight = 1
        easy_weight = 0.5
        loss_hm = self.cls_criterion(pred_heatmap, gt_heatmap)

        joint_num = pred_heatmap.shape[1]
        loss_offx = self.criterion(pred_offset_xy[:, :joint_num, :, :].mul(heatmap_mask), offset_x.mul(heatmap_mask))
        loss_offy = self.criterion(pred_offset_xy[:, joint_num:, :, :].mul(heatmap_mask), offset_y.mul(heatmap_mask))

        loss_hm = loss_hm * joint_mask
        loss_offx = loss_offx * joint_mask
        loss_offy = loss_offy * joint_mask

        if self.ohem:
            bs, joint_num = loss_hm.shape[:2]
            loss_hm = torch.reshape(loss_hm, (bs * joint_num, -1))
            loss_offx = torch.reshape(loss_offx, (bs * joint_num, -1))
            loss_offy = torch.reshape(loss_offy, (bs * joint_num, -1))

            hm_items = loss_hm.mean(dim=-1)
            offx_items = loss_offx.mean(dim=-1)
            offy_items = loss_offy.mean(dim=-1)
            sortids = torch.argsort(hm_items)
            topids = sortids[: (bs * joint_num // 3)]
            midids = sortids[(bs * joint_num // 3) : (bs * joint_num // 3 * 2)]
            bottomids = sortids[(bs * joint_num // 3 * 2) :]

            loss_hm = (
                (hm_items[topids] * easy_weight).mean()
                + (hm_items[midids] * mid_weight).mean()
                + (hm_items[bottomids] * hard_weight).mean()
            )
            loss_offx = (
                (offx_items[topids] * easy_weight).mean()
                + (offx_items[midids] * mid_weight).mean()
                + (offx_items[bottomids] * hard_weight).mean()
            )
            loss_offy = (
                (offy_items[topids] * easy_weight).mean()
                + (offy_items[midids] * mid_weight).mean()
                + (offy_items[bottomids] * hard_weight).mean()
            )

        return loss_hm, loss_offx, loss_offy
  
    def _compute_loss_with_heatmap(self, uv_heatmap, uv_off_xy, heatmap, heatmap_weight, offset_x, offset_y, joints_vis) -> Dict[str, torch.Tensor]:
        """compute loss"""
        batch_size = uv_heatmap.shape[0]
        joints_vis = (joints_vis.sum(dim=1) > 0).long()
        joint_mask = joints_vis.reshape(batch_size, 1, 1, 1).repeat(
            1, uv_heatmap.shape[1], uv_heatmap.shape[2], uv_heatmap.shape[3]
        )

        # 2d heatmap + depth loass
        loss_hm, loss_offx, loss_offy = self._loss(
            uv_heatmap, uv_off_xy, heatmap, joint_mask, heatmap_weight, offset_x, offset_y
        )

        outloss = dict()
        outloss["loss_uv_hm"] = self.heatmap_loss_weight * loss_hm
        outloss["loss_xy_offset"] = self.offset_loss_weight * (loss_offx + loss_offy)
        return outloss

    def forward_train(self, image, heatmap, offset_x, offset_y, heatmap_weight, joints_vis, **kwargs):
        output = self.backbone(image)   # x32,x16,x8,x4
        output = self.head(output)      # uv_heatmap, uv_off

        uv_heatmap, uv_off = output
        loss_output = \
            self._compute_loss_with_heatmap(
                uv_heatmap, uv_off, heatmap, heatmap_weight, offset_x, offset_y, joints_vis)
        return loss_output

    def get_max_pred_batch(self, batch_heatmaps, batch_offset_x, batch_offset_y):
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.max(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        for r in range(batch_size):
            for c in range(num_joints):
                dx = batch_offset_x[r, c, int(preds[r, c, 1]), int(preds[r, c, 0])]
                dy = batch_offset_y[r, c, int(preds[r, c, 1]), int(preds[r, c, 0])]
                preds[r, c, 0] += dx
                preds[r, c, 1] += dy

        return preds, maxvals

    def forward_test(self, image, **kwargs):
        image_h, image_w = image.shape[2:]
        output = self.backbone(image)   # x32,x16,x8,x4
        output = self.head(output)      # uv_heatmap, uv_off
  
        uv_heatmap, uv_off = output
        # convert to probability
        uv_heatmap = torch.sigmoid(uv_heatmap)
        heatmap_h, heatmap_w = uv_heatmap.shape[2:]
        joint_num = uv_heatmap.shape[1]
        offset_x, offset_y = uv_off[:, :joint_num, :, :], uv_off[:, joint_num:, :, :]
        preds = uv_heatmap.detach().cpu().numpy()
        offset_x = offset_x.detach().cpu().numpy()
        offset_y = offset_y.detach().cpu().numpy()

        preds, score = self.get_max_pred_batch(preds, offset_x, offset_y)
        preds[:,:, 0] = preds[:,:, 0] * (image_w/heatmap_w)
        preds[:,:, 1] = preds[:,:, 1] * (image_h/heatmap_h)
        pred_gt_hm = np.concatenate([preds, score], axis=2)
        results = {
            'pred_joints2d': [sample_joint2d for sample_joint2d in pred_gt_hm],
        }
        return results

    def onnx_export(self, image):
        output = self.backbone(image)   # x32,x16,x8,x4
        heatmap, offset = self.head(output) # uv_heatmap, uv_off
        heatmap = torch.sigmoid(heatmap)
        heatmap = F.max_pool2d(heatmap, 3, stride=1, padding=(3 - 1) // 2)
        return heatmap, offset