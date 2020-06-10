# -*- coding: UTF-8 -*-
# @Time    : 2019-06-17 17:26
# @File    : face_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.base import *
from antgo.task.task import *

default = {'AntFaceVROC': ('FaceVROC', 'FACE_VERIFICATION')}

class AntFaceVROC(AntMeasure):
  def __init__(self, task):
    super(AntFaceVROC, self).__init__(task, "FaceVROC")
    assert(task.task_type=='FACE_VERIFICATION')
    self.dist_type = getattr(self.task, 'dist_type', 'l2')

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    person_db = {}
    same_person = []
    diff_person = []

    for det_person, person_annotation in data:
      person_pair_id = person_annotation['pair_id']
      person_issame = person_annotation['same']
      det_person_id = person_annotation['id']

      if person_issame and (person_pair_id not in person_db and det_person_id not in person_db):
        same_person.append((det_person_id, person_pair_id))
      elif person_pair_id not in person_db and det_person_id not in person_db:
        diff_person.append((det_person_id, person_pair_id))

      person_db[det_person_id] = det_person

    dist_list = []
    label_list = []
    for person_a_id, person_b_id in same_person:
      person_a_feat = person_db[person_a_id]
      person_b_feat = person_db[person_b_id]

      if self.dist_type == 'l2':
        dist_list.append(np.sqrt(np.sum(np.power(person_a_feat-person_b_feat, 2.0))))
      else:
        dist_list.append(-np.dot(person_a_feat, person_b_feat))

      label_list.append(1)

    for person_a_id, person_b_id in diff_person:
      person_a_feat = person_db[person_a_id]
      person_b_feat = person_db[person_b_id]

      if self.dist_type == 'l2':
        dist_list.append(np.sqrt(np.sum(np.power(person_a_feat-person_b_feat, 2.0))))
      else:
        dist_list.append(-np.dot(person_a_feat, person_b_feat))

      label_list.append(0)

    actual = np.array(label_list).astype(np.int64)
    posterior = np.array(dist_list)
    num = actual.shape[0]
    positive_num = len(np.where(actual == 1)[0])
    negative_num = num - positive_num

    true_positive_num = 0.0
    false_positive_num = 0.0

    sorted_x = sorted(zip(posterior, range(len(posterior))))

    fpr = []
    tpr = []
    for i in range(len(sorted_x)):
        if actual[sorted_x[i][1]] == 1:
            true_positive_num += 1.0
        else:
            false_positive_num += 1.0

        # false positive rate
        negative_num = negative_num if negative_num > 0 else negative_num + 0.0000000001
        fp_rate = false_positive_num / float(negative_num)
        # true positive rate
        positive_num = positive_num if positive_num > 0 else positive_num + 0.0000000001
        tp_rate = true_positive_num / float(positive_num)

        fpr.append(fp_rate)
        tpr.append(tp_rate)

    trp_0dot01 = np.interp(0.01, fpr, tpr)
    trp_0dot001 = np.interp(0.001, fpr, tpr)

    return {'statistic': {'name': self.name,
                          'value': [{'name': 'TRP@0.01',
                                     'value': trp_0dot01,
                                     'type': 'SCALAR',
                                     'x': 'x',
                                     'y': 'y'},
                                    {'name': 'TRP@0.001',
                                      'value': trp_0dot001,
                                      'type': 'SCALAR',
                                      'x': 'x',
                                      'y': 'y'},
                                    {'name': 'ROC',
                                     'value': np.array([fpr,tpr]).transpose(),
                                     'type': 'CURVE',
                                     'x': 'FPR',
                                     'y': 'TRP'}]}}


class AntFaceIRank(AntMeasure):
  def __init__(self, task):
    super(AntFaceIRank, self).__init__(task, "FaceIRank")
    assert(task.task_type=='FACE_IDENTIFICATION')

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)
