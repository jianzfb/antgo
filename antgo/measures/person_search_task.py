# -*- coding: UTF-8 -*-
# @Time    : 2019-04-17 22:21
# @File    : person_search_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.base import *
from antgo.utils._bbox import bbox_overlaps
from antgo.task.task import *


default={'AntPersonSearchCMC': ('CMC', 'PERSON_SEARCH')}


class AntPersonSearchCMC(AntMeasure):
  def __init__(self, task):
    super(AntPersonSearchCMC, self).__init__(task, "CMC")
    assert(task.task_type == 'PERSON_SEARCH')

    self.support_rank_index = 0
    self.is_support_rank = True
    self.top_k = 1

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    bingo_num = 0
    pool_size = 0
    for det_predict, det_gt in data:
      # detected bbox
      det_bbox = det_predict['det-bbox']
      # l2 distance between det bbox and gt bbox
      det_score = np.array(det_predict['det-score'])
      det_sorted_inds = np.argsort(det_score, axis=1).tolist()

      gt_bbox = det_gt['bbox']
      gt_category = det_gt['category_id']
      overlaps = bbox_overlaps(
        np.ascontiguousarray(det_bbox, dtype=np.float),
        np.ascontiguousarray(gt_bbox, dtype=np.float))

      pool_size += len(gt_bbox)
      gtm = np.ones((len(gt_bbox))) * (-1)
      for dind, d in enumerate(det_bbox):
        best_match = -1
        iou_thres = 0.5
        for gind, g in enumerate(gt_bbox):
          if gtm[gind] >= 0:
            continue
          if overlaps[dind, gind] < iou_thres:
            continue

          iou_thres = overlaps[dind, gind]
          best_match = gind

        if best_match > -1:
          gtm[best_match] = dind
          if gt_category[best_match] in det_sorted_inds[dind][0:self.top_k]:
            bingo_num += 1

    cmc = float(bingo_num)/float(pool_size)
    return {'statistic': {'name': self.name,
                          'value': [{'name': 'CMC',
                                     'value': cmc,
                                     'type': 'SCALAR',
                                     'x': '',
                                     'y': ''},
                                   ]}}


if __name__ == '__main__':
  num = 20
  data = []
  for _ in range(num):
    x1 = np.random.random()
    y1 = np.random.random()

    x2 = x1 + np.random.random()
    y2 = y1 + np.random.random()

    det_predict = {'det-bbox': [[x1, y1, x2, y2]],
                   'det-score': np.random.random((1, 10)).tolist()}

    det_gt = {
      'bbox': [[x1,y1,x2,y2]]*10,
      'category_id':[3]*10
    }

    data.append((det_predict, det_gt))

  cmc = AntPersonSearchCMC(None)
  result = cmc.eva(data,None)
  pass