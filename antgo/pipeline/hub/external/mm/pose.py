# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 20:39
# @File    : pose.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from mmpose.apis import init_pose_model, inference_bottom_up_pose_model
import os
import sys
import subprocess


class Pose(object):
  def __init__(self, config_file, checkpoint_file, device='cpu'):
    if not config_file.endswith('.py'):
      config_file = f'{config_file}.py'
    config_name = config_file.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmpose')
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, config_file)
    if not os.path.exists(model_file):
      subprocess.check_call([sys.executable, '-m', 'mim', 'download', 'mmpose', '--config', config_name, '--dest', model_folder])

    if checkpoint_file is None:
      for f in os.listdir(model_folder):
        if f.startswith(config_name) and f.endswith('.pth'):
          checkpoint_file = f
          break
    assert(checkpoint_file is not None)

    config_file = os.path.join(model_folder, config_file)
    checkpoint_file = os.path.join(model_folder, checkpoint_file)

    self.model = init_pose_model(config_file, checkpoint_file, device=device)  # or device='cuda:0'

  def __call__(self, *args, **kwargs):
    pose_results, _ = inference_bottom_up_pose_model(self.model, args[0])

    # label
    return pose_results[0]
