from ..runner.builder import *
import copy
import json
import numpy as np
import time


@MEASURES.register_module()
class OKS(object):
    def __init__(self, sigmas=None, weights=None) -> None:
        self.sigmas = None
        if sigmas is not None:
            self.sigmas = np.array(sigmas)
        self.weights = np.array(weights)

    def keys(self):
        # 约束使用此评估方法，需要具体的关键字信息
        return {'pred': ['pred_joints2d'], 'gt': ['joints2d', 'bboxes', 'joints_vis']}

    def __call__(self, preds, gts):
        oks_list = []
        for pred, gt in zip(preds,gts):
            # 预测信息
            pred_joints2d = pred['pred_joints2d']   # Nx2

            # GT信息
            gt_joints2d = gt['joints2d']            # Nx2            
            gt_bboxes = gt['bboxes']                # 4
            gt_joints_vis = gt['joints_vis']        # N
            if np.sum(gt_joints_vis) == 0:
                continue

            if gt_joints2d.ndim == 2:
                # 单目标预测
                pred_joints2d = pred_joints2d[np.newaxis, :, :]
                gt_joints2d = gt_joints2d[np.newaxis, :, :]
                gt_bboxes = gt_bboxes[np.newaxis, :]
                gt_joints_vis = gt_joints_vis[np.newaxis, :]

            # area: Nx1
            area = (gt_bboxes[:,2] - gt_bboxes[:,0])*(gt_bboxes[:,3] - gt_bboxes[:,1])
            area = area[:, np.newaxis]

            # dist: Nx33
            dist = np.sqrt(np.sum(np.power(pred_joints2d[:,:,:2]-gt_joints2d, 2.0), -1))
            dist = dist / np.power(area, 0.5).clip(min=1e-6)
            if self.sigmas is not None:
                sigmas = self.sigmas.reshape(*((1, ) * (dist.ndim - 1)), -1)
                dist = dist / (sigmas * 2)

            dist = dist * gt_joints_vis

            if self.weights is None or self.weights.ndim != 2:
                if self.weights is None:
                    self.weights = np.ones((1, dist.shape[-1]))
                else:
                    self.weights = self.weights.reshape(*((1, ) * (dist.ndim - 1)), -1)

                self.weights = self.weights / np.sum(self.weights, -1).clip(min=1e-6)

            oks_v = (np.exp(-np.power(dist, 2) / 2) * self.weights).sum(-1)
            oks_list.append(np.mean(oks_v))

        oks_err = float(np.mean(oks_list))
        print()
        print("======== OKS Acc: %f ========" % (oks_err))

        error_info = {
            'oks': oks_err
        }
        return error_info
