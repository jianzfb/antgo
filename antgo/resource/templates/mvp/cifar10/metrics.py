from antgo.framework.helper.runner.builder import *
import numpy as np


@MEASURES.register_module()
class AccuracyEval(object):
    def __init__(self, topk=(1, ), thrs=0.) -> None:
        self.topk = topk
        self.thrs = thrs

    def keys(self):
        # 约束使用此评估方法，需要具体的关键字信息
        return {'pred': ['pred'], 'gt': ['label']}

    def __call__(self, preds, gts):
        preds_reformat = [pred['pred'] for pred in preds]
        gts_reformat = [gt['label'] for gt in gts]
        preds = np.stack(preds_reformat, 0)
        gts = np.array(gts_reformat).astype(np.int32)

        maxk = max(self.topk)
        num = preds.shape[0]

        static_inds = np.indices((num, maxk))[0]
        pred_label = preds.argpartition(-maxk, axis=1)[:, -maxk:]
        pred_score = preds[static_inds, pred_label]

        sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
        pred_label = pred_label[static_inds, sort_inds]
        pred_score = pred_score[static_inds, sort_inds]

        eval_values = {}
        for k in self.topk:
            correct_k = pred_label[:, :k] == gts.reshape(-1, 1)
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > self.thrs)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            eval_values[f'top_{k}'] = _correct_k.sum() * 100. / num

        return eval_values