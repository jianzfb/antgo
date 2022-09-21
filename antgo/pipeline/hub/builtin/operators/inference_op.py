# -*- coding: UTF-8 -*-
# @Time    : 2022/9/21 22:42
# @File    : inference_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.reader import *
from antgo.framework.helper.dataset import PIPELINES
from antgo.framework.helper.utils import build_from_cfg
from antgo.framework.helper.parallel.collate import collate
from antgo.framework.helper.cnn.utils import fuse_conv_bn
import torch
import cv2
import json
import copy
import numpy as np


@register
class inference_op(object):
  def __init__(self, model_name=None, cfg=None, checkpoint='', device='cpu', is_fuse_conv_bn=False, output=None) -> None:
    super().__init__()
    self.device = device
    self.cfg = cfg
    if type(cfg) != str:
      self.cfg = Config.fromstring(json.dumps(cfg), '.json')

    # build model
    self.model = build_model(self.cfg.model)
    if checkpoint == '':
      checkpoint = self.cfg.get('checkpoint', checkpoint)
    checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')

    if is_fuse_conv_bn:
      print('use fuse conv_bn')
      self.model = fuse_conv_bn(self.model)

    self.model = build_dp(self.model, self.device, device_ids=self.cfg.gpu_ids)

    # build data reader
    pipeline = self.cfg.data.test.pipeline
    inputs_def = self.cfg.data.test.inputs_def
    self.pipeline = []
    self.pipeline_types = []
    for transform in pipeline:
      if isinstance(transform, dict):
        self.pipeline_types.append(transform['type'])
        transform = build_from_cfg(transform, PIPELINES)
        self.pipeline.append(transform)
      else:
        raise TypeError('pipeline must be a dict')

    self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None
    self._output = output

  def _arrange(self, sample, fields):
    if type(fields[0]) == list or type(fields[0]) == tuple:
      warp_ins = []
      for field in fields:
        one_ins = {}
        for ff in field:
          one_ins[ff] = sample[ff]

        warp_ins.append(one_ins)
      return warp_ins

    warp_ins = {}
    for field in fields:
      warp_ins[field] = sample[field]

    return warp_ins

  def __call__(self, image):
    image = image[:, :, 0]
    sample = {
      'image': image,
      'image_metas': {
        'image_shape': image.shape,
        'scale_factor': (1, 1, 1, 1),
        'image_file': '',
        'transform_matrix': np.eye(3),
      }
    }

    # transform
    for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
      try:
        sample = transform(sample)
      except:
        print(f'transform error{transform_type}')

    # arange warp
    sample = self._arrange(sample, self._fields)
    sample = collate([sample])
    sample.update({
      'return_loss': False
    })

    # model run
    self.model.eval()
    result = {}
    with torch.no_grad():
      result = self.model(**sample)

    if self._output is None:
      self._output = result.keys()
    out = []
    for k in self._output:
      out.append(result[k][0].detach().cpu().numpy())
    return tuple(out)




