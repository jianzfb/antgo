# encoding=utf-8
# @Time    : 17-7-12
# @File    : segmentation_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *
from antgo.dataflow.common import *

__all__ = {'AntPixelAccuracySeg': ('PixelAccuracy', 'SEGMENTATION'),
           'AntMeanAccuracySeg': ('MeanAccuracy', 'SEGMENTATION'),
           'AntMeanIOUSeg': ('MeanIOU', 'SEGMENTATION'),
           'AntFrequencyWeightedIOUSeg': ('FrequencyWeightedIOU', 'SEGMENTATION'),
           'AntMeanIOUBoundary': ('MeanIOUBoundary', 'SEGMENTATION'),
           }
class AntPixelAccuracySeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: \sum_i{n_{ii}}/\sum_i t_i
    super(AntPixelAccuracySeg, self).__init__(task, 'PixelAccuracy')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    sum_nii = np.zeros((classes_num))
    sum_ti = np.zeros((classes_num))

    if label is not None:
        data = zip(data, label)

    sample_scores = []
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
      
      gt_labels = set(gt.flatten())
      for l in gt_labels:
        l = int(l)
        # if l == 0:
        #   continue

        p = np.where(gt == l)
        sum_ti[l] += len(p[0])

        predicted_l = predict[p]
        nii = len(np.where(predicted_l == l)[0])
        sum_nii[l] += nii
        
        if id is not None:
          sample_scores.append({'id': id, 'score': float(nii) / float(len(p[0])), 'category': l})

    val = np.sum(sum_nii) / (np.sum(sum_ti) + 1e-6)

    # # temparary test statistic quantity
    # # 1.step curve statistic
    # curve_measure = {'name': 'MAP', 'value': [23.0, 11.0, 12.0], 'type': 'SCALAR', 'x': 'class',
    #                                         'y': 'Mean Average Precision'}
    #
    # # 2.step histogram statistic
    # histogram_measure = {'name': 'ROC', 'value': [[[0, 0], [1, 1], [2, 3]],
    #                                                                      [[0, 3], [2, 5], [3, 0]]],
    #                                             'type': 'CURVE', 'x': 'FP', 'y': 'TP',
    #                                             'legend': ['class-0', 'class-1']}
    #
    # # 3.step matrix statistic
    # matrix_measure = {'name': 'ccmm', 'value': (np.ones((3, 4)) * 3).tolist(), 'type': 'MATRIX',
    #                                         'x': ['a', 'b', 'c', 'd'], 'y': ['x', 'y', 'z']}
    #
    # # 4.step image statistic
    # random_img = np.random.random((100, 100))
    # random_img = random_img * 255
    # random_img = random_img.astype(np.uint8)
    # image_measure = {'name': 'image', 'value': random_img, 'type': 'IMAGE'}

    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]},
            'info': sample_scores}


class AntMeanAccuracySeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: (1/n_{cl}) / \sum_i n_{ii}/t_i
    super(AntMeanAccuracySeg, self).__init__(task, 'MeanAccuracy')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    sum_nii = np.zeros((classes_num))
    sum_ti = np.zeros((classes_num))

    if label is not None:
      data = zip(data, label)

    sample_scores = []
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
        
      gt_labels = set(gt.flatten())
      for l in gt_labels:
        l = int(l)
        # if l == 0:
        #   continue

        p = np.where(gt == l)
        sum_ti[l] += len(p[0])

        predicted_l = predict[p]
        nii = len(np.where(predicted_l == l)[0])
        sum_nii[l] += nii
        
        if id is not None:
          sample_scores.append({'id': id, 'score': float(nii) / float(len(p[0])), 'category': l})

    val = np.mean(sum_nii / (sum_ti + 1e-6))
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]},
            'info': sample_scores}


class AntMeanIOUSeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUSeg, self).__init__(task, 'MeanIOU')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    sum_nii = np.zeros((classes_num))
    sum_ti = np.zeros((classes_num))
    sum_ji = np.zeros((classes_num))

    if label is not None:
      data = zip(data, label)

    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
  
      gt_labels = set(gt.flatten())
      for l in gt_labels:
        l = int(l)
        # if l == 0:
        #   continue
        p = np.where(gt == l)
        sum_ti[l] += len(p[0])

        predicted_l = predict[p]
        nii = len(np.where(predicted_l == l)[0])
        sum_nii[l] += nii

        sum_ji[l] += len(np.where(predict == l)[0])
        
    val = np.mean(sum_nii / (sum_ti + sum_ji - sum_nii + 1e-6))
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'}]}}


class AntFrequencyWeightedIOUSeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: (\sum_kt_k)^{-1} / \sum_i t_in_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntFrequencyWeightedIOUSeg, self).__init__(task, 'FrequencyWeightedIOU')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    sum_nii = np.zeros((classes_num))
    sum_ti = np.zeros((classes_num))
    sum_ji = np.zeros((classes_num))

    if label is not None:
        data = zip(data, label)

    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
  
      gt_labels = set(gt.flatten())
      for l in gt_labels:
        l = int(l)
        # if l == 0:
        #     continue
        p = np.where(gt == l)
        sum_ti[l] += len(p[0])

        predicted_l = predict[p]
        nii = len(np.where(predicted_l == l)[0])
        sum_nii[l] += nii

        sum_ji[l] += len(np.where(predict == l)[0])

    val = np.sum(sum_ti * sum_nii / (sum_ti + sum_ji - sum_nii + 1e-6)) / (np.sum(sum_ti) + 1e-6)
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]}}


class AntMeanIOUBoundary(AntMeasure):
  def __init__(self, task):
    # paper: Liang-Chieh Chen, etc. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUBoundary, self).__init__(task, 'MeanIOUBoundary')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    sum_nii = np.zeros((classes_num))
    sum_ti = np.zeros((classes_num))
    sum_ji = np.zeros((classes_num))

    trimap_width = int(getattr(self.task, 'trimap_width', 3))
    offset_x, offset_y = np.meshgrid(np.arange(-trimap_width, trimap_width + 1),
                                     np.arange(-trimap_width, trimap_width + 1))
    offset = np.column_stack((offset_x.flatten(), offset_y.flatten()))

    if label is not None:
        data = zip(data, label)
    
    sample_scores = []
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
      
      gt_labels = set(gt.flatten())
      rows, cols = gt.shape[:2]
      # generate trimap for objects (predict)
      predict_boundary = np.where((predict[0:-1, 0:-1] - predict[1:, 1:]) != 0)
      predict_boundary = np.column_stack(predict_boundary)
      predict_band_boundary = np.expand_dims(predict_boundary, 1) + offset
      predict_band_boundary = predict_band_boundary.reshape(-1, 2)
      index = np.where((predict_band_boundary[:, 0] > 0) &
                       (predict_band_boundary[:, 1] > 0) &
                       (predict_band_boundary[:, 0] < rows) &
                       (predict_band_boundary[:, 1] < cols))
      predict_trimap = np.zeros((rows, cols), dtype=np.int32)
      predict_trimap[predict_band_boundary[index, 0], predict_band_boundary[index, 1]] = \
          predict[predict_band_boundary[index, 0], predict_band_boundary[index, 1]]

      for l in gt_labels:
        l = int(l)
        # if l == 0:
        #   continue
        # generate trimap for object (gt)
        obj_map = np.zeros((rows, cols), dtype=np.uint32)
        obj_map[np.where(gt == l)] = 1

        gt_boundary = np.where((obj_map[0:-1, 0:-1] - obj_map[1:, 1:]) != 0)
        gt_boundary = np.column_stack(gt_boundary)
        gt_band_boundary = np.expand_dims(gt_boundary, 1) + offset
        gt_band_boundary = gt_band_boundary.reshape(-1, 2)
        gt_band_index = np.where((gt_band_boundary[:, 0] > 0) &
                                 (gt_band_boundary[:, 1] > 0) &
                                 (gt_band_boundary[:, 0] < rows) &
                                 (gt_band_boundary[:, 1] < cols))

        # trimap = np.zeros((rows, cols), dtype=np.int32)
        # trimap[gt_band_boundary[gt_band_index,0],gt_band_boundary[gt_band_index,1]] = 1
        # cv2.imshow("DD", (trimap * 255).astype(np.uint8))
        # cv2.imshow("ZZ", (gt * 255).astype(np.uint8))
        # cv2.waitKey(0)

        sum_ti[l] += len(gt_band_index[0])

        predicted_l = predict_trimap[gt_band_boundary[gt_band_index, 0], gt_band_boundary[gt_band_index, 1]]
        nii = len(np.where(predicted_l == l)[0])
        sum_nii[l] += nii
        
        vv = len(np.where(predict_trimap == l)[0])
        sum_ji[l] += vv
        
        if id is not None:
          sample_scores.append({'id': id, 'score': float(nii) / float(len(gt_band_index[0]) + vv - nii), 'category': l})

    val = np.mean(sum_nii / (sum_ti + sum_ji - sum_nii + 1e-6))
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'}]},
            'info': sample_scores}