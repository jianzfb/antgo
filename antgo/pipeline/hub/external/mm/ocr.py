# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 20:39
# @File    : ocr.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import copy
import os
from mmocr.apis import MMOCRInferencer


class Ocr(object):
  def __init__(self, det_config='dbnetpp', recog_config='SAR', device='cuda:0'):
      self.det_config = det_config
      self.recog_config = recog_config
      self.device = device
      self.ocr_infer = MMOCRInferencer(det=self.det_config, rec=self.recog_config, device=self.device)

  def __call__(self, *args, **kwargs):
    result = self.ocr_infer(args[0], return_vis=False)
    if len(result['predictions']) == 0:
        # det_polygons, det_scores, rec_texts, rec_scores
        return None, None, None, None

    predictions = result['predictions'][0]
    det_polygons = None
    det_scores = None
    if 'det_polygons' in predictions:
        det_polygons = predictions['det_polygons']
    if 'det_scores' in predictions:
        det_scores = predictions['det_scores']

    rec_texts = None
    rec_scores = None
    if 'rec_texts' in predictions:
        rec_texts = predictions['rec_texts']
    if 'rec_scores' in predictions:
        rec_scores = predictions['rec_scores']

    return det_polygons, det_scores, rec_texts, rec_scores
