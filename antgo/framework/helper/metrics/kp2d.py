from ..runner.builder import *
import copy
import json
import numpy as np
import time


@MEASURES.register_module()
class OKS(object):
    def __init__(self, sigma=0.1) -> None:
        self.sigma = sigma

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

            gt_joints_vis = np.array(gt_joints_vis).reshape(-1, 1)
            x0,y0,x1,y1 = gt_bboxes            
            joints_2d_dist = np.sum(np.power(pred_joints2d[:,:2]-gt_joints2d, 2.0), -1,keepdims=True)
            Z = 2.0 * (x1-x0)*(y1-y0) * self.sigma

            oks_v = np.sum(np.exp(-joints_2d_dist/Z)*gt_joints_vis)/np.sum(gt_joints_vis)
            oks_list.append(oks_v)

        oks_err = float(np.mean(oks_list))
        print()
        print("======== OKS Acc: %f ========" % (oks_err))

        error_info = {
            'oks': oks_err
        }
        return error_info
