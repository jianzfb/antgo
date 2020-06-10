# -*- coding: UTF-8 -*-
# @Time    : 18-3-19
# @File    : ali_fashion_landmark_ne.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
import numpy as np

default = {'AntALIFashionLandmarkNE': ('FashionLandmarkNE', 'LANDMARK')}
class AntALIFashionLandmarkNE(AntMeasure):
  def __init__(self, task):
    super(AntALIFashionLandmarkNE, self).__init__(task, 'FashionLandmarkNE')
    assert(task.task_type == 'LANDMARK')
    self.is_support_rank = True
    self.is_inverse = True
    
  def eva(self, data, label):
    '''
    
    :param data:
    :param label:
    :return:
    '''
    
    if label is not None:
      data = zip(data, label)
    
    measure_val = 0.0
    sample_scores = []
    for predict, gt in data:
      predict_landmarks, predict_category = predict

      gt_landmarks = gt['landmark']
      gt_category = gt['category']
      id = gt['id']

      assert(predict_category == gt_category)
      
      # normalized parameter
      sk = 0.0
      if gt_category in ['blouse', 'outwear', 'dress']:
        # 衬衫，外套，连衣裙
        if gt_landmarks[5][4] in [0, 1] and gt_landmarks[6][4] in [0, 1]:
          armpit_left_px = gt_landmarks[5][2]
          armpit_left_py = gt_landmarks[5][3]

          armpit_right_px = gt_landmarks[6][2]
          armpit_right_py = gt_landmarks[6][3]

          sk = np.sqrt(np.power(armpit_left_px - armpit_right_px,2)+
                       np.power(armpit_left_py - armpit_right_py,2)) + 1e-6
      else:
        # 裤子，半身裙
        if gt_landmarks[15][4] in [0, 1] and gt_landmarks[16][4] in [0, 1]:
          waistband_left_px = gt_landmarks[15][2]
          waistband_left_py = gt_landmarks[15][3]

          waistband_right_px = gt_landmarks[16][2]
          waistband_right_py = gt_landmarks[16][3]

          sk = np.sqrt(np.power(waistband_left_px - waistband_right_px, 2) +
                       np.power(waistband_left_py - waistband_right_py, 2)) + 1e-6

      if sk < 1e-5:
        continue

      denominator = 1e-6
      numerator = 0.0
      for predict_landmark, gt_landmark in zip(predict_landmarks, gt_landmarks):
        predict_x, predict_y, predict_visible = predict_landmark
        gt_x, gt_y, gt_visible = gt_landmark[2:]
        
        if gt_visible == 1:
          denominator += 1
          numerator += (np.sqrt(np.power(predict_x-gt_x,2.0) + np.power(predict_y-gt_y,2.0))) / sk if predict_visible == 1 else 1.0

      score = numerator / denominator
      sample_scores.append({'id': id, 'score': score, 'category': gt_category})
      measure_val += score
    
    measure_val /= float(len(sample_scores))
    return {'statistic': {'name': self.name,
                          'value':[{'name': self.name, 'value': measure_val, 'type': 'SCALAR'}]},
            'info': sample_scores}