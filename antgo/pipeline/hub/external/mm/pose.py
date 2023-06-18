# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 20:39
# @File    : pose.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from mmpose.apis import init_model, inference_topdown, inference_bottomup, Pose2DInferencer
import os
import sys
import subprocess


class Pose(object):
  def __init__(self, config_file, checkpoint_file=None, is_bottomup=True, device='cuda:0'):
    if not config_file.endswith('.py'):
      config_file = f'{config_file}.py'
    config_name = config_file.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmpose', config_name)
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, config_file)
    if not os.path.exists(model_file):
      subprocess.check_call(['mim', 'download', 'mmpose', '--config', config_name, '--dest', model_folder])

    if checkpoint_file is None:
      for f in os.listdir(model_folder):
        if f.endswith('.pth'):
          checkpoint_file = f
          break
    assert(checkpoint_file is not None)

    config_file = os.path.join(model_folder, config_file)
    checkpoint_file = os.path.join(model_folder, checkpoint_file)

    self.model = init_model(config_file, checkpoint_file, device=device)  # or device='cuda:0'
    self.is_bottomup = is_bottomup

  def __call__(self, *args, **kwargs):
    pose_info = None
    if self.is_bottomup:
      pose_info, = inference_bottomup(self.model, args[0])
    else:
      pose_info, = inference_topdown(self.model, args[0])

    keypoints = pose_info.pred_instances.keypoints
    keypoints_scores = pose_info.pred_instances.keypoint_scores
    bboxes = pose_info.pred_instances.bboxes
    return keypoints, keypoints_scores, bboxes
