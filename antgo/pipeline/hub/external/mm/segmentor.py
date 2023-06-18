# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 20:26
# @File    : segmentor.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import mmcv
from mmseg.apis import inference_model, init_model
import os
import subprocess
import sys


class Segmentor(object):
  def __init__(self, config_file, checkpoint_file=None, device='cuda:0'):
    if not config_file.endswith('.py'):
      config_file = f'{config_file}.py'
    config_name = config_file.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmsegmentation', config_name)
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, config_file)
    if not os.path.exists(model_file):
      subprocess.check_call(['mim', 'download', 'mmsegmentation', '--config', config_name, '--dest', model_folder])

    if checkpoint_file is None:
      for f in os.listdir(model_folder):
        if f.endswith('.pth'):
          checkpoint_file = f
          break
    assert(checkpoint_file is not None)

    config_file = os.path.join(model_folder, config_file)
    checkpoint_file = os.path.join(model_folder, checkpoint_file)
    self.model = init_model(config_file, checkpoint_file, device=device)

  def __call__(self, *args, **kwargs):
    result = inference_model(self.model, args[0])

    # label
    return result.pred_sem_seg.data.to('cpu').numpy()[0]
