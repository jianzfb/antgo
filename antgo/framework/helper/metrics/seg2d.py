from ..runner.builder import *
import copy
import json
import numpy as np
import time


@MEASURES.register_module()
class SegMIOU(object):
    def __init__(self, class_num, ignore_val=255) -> None:
        self.ignore_val = ignore_val
        self.class_num = class_num
    
    def keys(self):
        return {'pred': ['pred_segments'], 'gt':['segments']}
    
    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.class_num) & (label_true != self.ignore_val)

        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.class_num * label_true[mask].astype(np.int) +
            label_pred[mask].astype(np.int), minlength=self.class_num ** 2).reshape(self.class_num, self.class_num)
        return hist

    def __call__(self, preds, gts):
        confusion = np.zeros((self.class_num, self.class_num))
        for pred, gt in zip(preds, gts):
            sample_confusion = self._fast_hist(np.array(pred['pred_segments']).flatten(), np.array(gt['segments']).flatten())
            confusion += sample_confusion

        iou = np.diag(confusion) / np.maximum(1.0,confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
        miou_val = float(np.nanmean(iou))

        print()
        print("======== MIOU Acc: %f ========" % (miou_val))
        error_info = {
            'miou': miou_val
        }
        return error_info