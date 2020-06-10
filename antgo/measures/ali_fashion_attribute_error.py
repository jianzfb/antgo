# -*- coding: UTF-8 -*-
# @Time : 26/03/2018
# @File : ali_fashion_attribute_error.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
import numpy as np


default = {'AntALIFashionAttributeError': ('FashionAttributeError', 'CLASSIFICATION')}
class AntALIFashionAttributeError(AntMeasure):
  def __init__(self, task):
    super(AntALIFashionAttributeError, self).__init__('FashionAttributeError')
    assert(task.task_type == 'CLASSIFICATION')
    self.is_support_rank = True

    self.cloth_attribs = ['skirt_length_labels',
                          'coat_length_labels',
                          'collar_design_labels',
                          'lapel_design_labels',
                          'neck_design_labels',
                          'neckline_design_labels',
                          'pant_length_labels',
                          'sleeve_length_labels']

  def eva(self, data, label):
    '''

    :param data:
    :param label:
    :return:
    '''
    # traverse all attrib
    mean_ap = []
    for cloth_attrib in self.cloth_attribs:
      max_attr_value_prob_list = []
      zip_data = data if label is None else zip(data, label)
      for predict, gt in zip_data:
        # filter
        if gt['category'] != cloth_attrib:
          continue

        predict_prob, predict_cloth_attrib = predict
        assert(predict_cloth_attrib == cloth_attrib)

        max_attr_value_prob = np.max(predict_prob)
        max_attr_value_index = np.argmax(predict_prob)

        if gt['category_id'][max_attr_value_index] == 1:
          max_attr_value_prob_list.append((max_attr_value_prob, 1))
        elif gt['category_id'][max_attr_value_index] == 0:
          max_attr_value_prob_list.append((max_attr_value_prob, 0))

      # compute AP
      sorted_prob = sorted(max_attr_value_prob_list, key=lambda x: -x[0])
      predict_correct_count = 0
      predict_count = 0
      val_list = []
      for prob in sorted_prob:
        if prob[1] == 1:
          predict_correct_count += 1
          predict_count += 1
        else:
          predict_count += 1

        val = float(predict_correct_count) / float(predict_count)
        val_list.append(val)

      if len(val_list) == 0:
        print('sdf')
      ap = np.mean(val_list)
      mean_ap.append(ap)

    mean_ap_val = np.mean(mean_ap)
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': mean_ap_val, 'type': 'SCALAR'}]}}