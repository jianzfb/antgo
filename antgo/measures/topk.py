from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
import numpy as np

class AntTopK(AntMeasure):
  def __init__(self, task):
    super(AntTopK, self).__init__(task, 'ACCURACY')
    assert(task.task_type == 'CLASSIFICATION')
    self.is_support_rank = True
    self.larger_is_better = 1
    self.topk = (1,5)

def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    maxk = max(self.topk)
    res = [0 for _ in range(len(self.topk))]
    sample_scores = []
    for predict, gt in data:
        # predict: probability
        # gt: {'category_id': label, 'id': ..}
        id = gt['id']
        gt_label = gt['category_id']
        top_idx = predict.argsort()[-maxk:][::-1]

        correct = top_idx == gt_label
        sample_score = []
        for i, k in enumerate(self.topk):
            correct_k = np.sum(correct[:k].reshape(-1).astype(np.float32))
            sample_score[i] = correct_k
            res[i] += correct_k

        sample_scores.append({'id': id, 'score': sample_score[0], 'category': gt_label})
    
    totle_num = len(sample_scores)
    res[0] = res[0] / totle_num
    res[1] = res[1] / totle_num

    return {
      'statistic': {
        'name': self.name,
        'value': [
          {'name': 'top_1', 'value': float(res[0]), 'type': 'SCALAR'},
          {'name': 'top_5', 'value': float(res[1]), 'type': 'SCALAR'},
        ]
      },
      'info': sample_scores
    }