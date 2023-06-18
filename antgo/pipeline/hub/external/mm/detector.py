# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 13:53
# @File    : detector.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
import subprocess
import sys


class Detector(object):
  def __init__(self, config_file, checkpoint_file=None, thres=0.4, device='cuda:0'):
    self.thres = thres
    if not config_file.endswith('.py'):
      config_file = f'{config_file}.py'
    config_name = config_file.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmdetection', config_name)
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, config_file)
    if not os.path.exists(model_file):
      subprocess.check_call(['mim', 'download', 'mmdet', '--config', config_name, '--dest', model_folder])

    if checkpoint_file is None:
      for f in os.listdir(model_folder):
        if f.endswith('.pth'):
          checkpoint_file = f
          break
    assert(checkpoint_file is not None)

    config_file = os.path.join(model_folder, config_file)
    checkpoint_file = os.path.join(model_folder, checkpoint_file)
    self.model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'

  def __call__(self, *args, **kwargs):
    result = inference_detector(self.model, args[0])
    pred_instances_score = result.pred_instances.scores.to('cpu').numpy()
    pred_instances_label = result.pred_instances.labels.to('cpu').numpy()
    pred_instances_bbox = result.pred_instances.bboxes.to('cpu').numpy()
    valid_pos = np.where(pred_instances_score > self.thres)
    valid_pred_bbox = pred_instances_bbox[valid_pos]
    valid_pred_score = pred_instances_score[valid_pos]
    valid_pred_label = pred_instances_label[valid_pos]
    valid_pred_bbox = np.concatenate([valid_pred_bbox, valid_pred_score.reshape(-1, 1)], -1)
    return valid_pred_bbox, valid_pred_label

