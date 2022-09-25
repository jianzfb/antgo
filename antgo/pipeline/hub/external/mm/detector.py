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
  def __init__(self, config_file, checkpoint_file=None, device='cpu'):
    if not config_file.endswith('.py'):
      config_file = f'{config_file}.py'
    config_name = config_file.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmdetection')
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, config_file)
    if not os.path.exists(model_file):
      subprocess.check_call([sys.executable, '-m', 'mim', 'download', 'mmdet', '--config', config_name, '--dest', model_folder])

    if checkpoint_file is None:
      for f in os.listdir(model_folder):
        if f.startswith(config_name) and f.endswith('.pth'):
          checkpoint_file = f
          break
    assert(checkpoint_file is not None)

    config_file = os.path.join(model_folder, config_file)
    checkpoint_file = os.path.join(model_folder, checkpoint_file)
    self.model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'

  def __call__(self, *args, **kwargs):
    result = inference_detector(self.model, args[0])

    bbox = []
    label = []
    for cls_i, det_result in enumerate(result):
      if det_result.size > 0:
        det_num = det_result.shape[0]
        bbox.append(det_result)
        label.append(np.ones((det_num)) * (cls_i+1))

    if len(bbox) == 0:
      bbox = np.empty([0, 5])
      label = np.empty([0])
    else:
      bbox = np.concatenate(bbox,0)
      label = np.concatenate(label, 0)
    return bbox, label

