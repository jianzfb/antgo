# -*- coding: UTF-8 -*-
# @Time    : 17-12-29
# @File    : pck.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *


class AntPCK(AntMeasure):
  def __init__(self, task):
    super(AntPCK, self).__init__(task, 'PCK')
    assert(task.task_type == 'LANDMARK')
    
    self.joints = self.task.joints
    self.joints_normalizer = self.task.joints_normalizer
    self.is_support_rank = True

  def pcki(self, joint_id, gtJ, prJ, idlh=3, idrs=12):
    """ Compute PCK accuracy on a given joint
    Args:
      joint_id	: Index of the joint considered
      gtJ			: Ground Truth Joint
      prJ			: Predicted Joint
      idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
      idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
    Returns:
      (float) NORMALIZED L2 ERROR
    """
    return np.linalg.norm(gtJ[joint_id] - prJ[joint_id][::-1]) / np.linalg.norm(gtJ[idlh] - gtJ[idrs])

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)
    
    pck_ratio = [[] for _ in range(len(self.joints))]
    pck_flatten_ratio = []
    pck_id = []
    sample_scores = []
    for landmark_predict, gt in data:
      id = gt['id']
      landmark_gt = np.array(gt['landmark'])
      landmark_weight = np.ones((len(self.joints)))
      if len(landmark_gt.shape) == 1:
        landmark_gt = landmark_gt.reshape((-1, 2))
      for i in range(len(self.joints)):
        if landmark_gt[i, 0] < 0 or landmark_gt[i, 1] < 0:
          landmark_weight[i] = 0
      
      landmark_predict = np.array(landmark_predict)
      if len(landmark_predict) == 1:
        landmark_predict = landmark_predict.reshape((-1, 2))
      
      # landmark predict shape (joints, 2)
      # landmark gt shape(joints, 2)
      offset_s = len(pck_flatten_ratio)
      for joint_i in range(len(self.joints)):
        if landmark_weight[joint_i] > 0:
          value = self.pcki(joint_i,
                            landmark_gt,
                            landmark_predict,
                            self.joints_normalizer[0],
                            self.joints_normalizer[1])
          pck_ratio[joint_i].append(value)
          pck_flatten_ratio.append(value)
          pck_id[joint_i].append(id)
      
      offset_e = len(pck_flatten_ratio)
      sample_pck = pck_flatten_ratio[offset_s:offset_e]
      sample_scores.append({'id': id, 'score': float(np.mean(sample_pck))})
    
    pck_total_ratio = float(np.mean(pck_flatten_ratio))
    return {'statistic': {'name': self.name,
                          'value':[{'name': self.name, 'value': pck_total_ratio, 'type': 'SCALAR'}]},
            'info': sample_scores}