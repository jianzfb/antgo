# -*- coding: UTF-8 -*-
# @Time    : 2022/9/25 21:27
# @File    : restoration.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os

import mmcv
import torch

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.utils import modify_args
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose



def _restoration_inference(model, img, ref=None):
  """Inference image with the model.

  Args:
      model (nn.Module): The loaded model.
      img (str): File path of input image.
      ref (str | None): File path of reference image. Default: None.

  Returns:
      Tensor: The predicted restoration result.
  """
  cfg = model.cfg
  device = next(model.parameters()).device  # model device
  # remove gt from test_pipeline
  keys_to_remove = ['gt', 'gt_path', 'lq_path']
  for key in keys_to_remove:
    for pipeline in list(cfg.test_pipeline):
      if 'key' in pipeline and key == pipeline['key']:
        cfg.test_pipeline.remove(pipeline)
      if 'keys' in pipeline and key in pipeline['keys']:
        pipeline['keys'].remove(key)
        if len(pipeline['keys']) == 0:
          cfg.test_pipeline.remove(pipeline)
      if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
        pipeline['meta_keys'].remove(key)

  for pipeline in list(cfg.test_pipeline):
    if pipeline['type'] == 'LoadImageFromFile':
      cfg.test_pipeline.remove(pipeline)

  # build the data pipeline
  test_pipeline = Compose(cfg.test_pipeline)
  # prepare data
  if ref:  # Ref-SR
    data = dict(lq=img, ref=ref)
  else:  # SISR
    data = dict(lq=img)
  data = test_pipeline(data)
  data = collate([data], samples_per_gpu=1)
  if 'cuda' in str(device):
    data = scatter(data, [device])[0]
  # forward the model
  with torch.no_grad():
    result = model(test_mode=True, **data)

  return result['output']


class Restoration(object):
  def __init__(self, config_file, checkpoint_file, device='cpu'):
    self.model = init_model(config_file, checkpoint_file, device=device)

  def __call__(self, *args, **kwargs):
    img = None
    ref = None
    if len(args) == 2:
      img, ref = args
    else:
      img = args[0]

    result = _restoration_inference(self.model, img, ref)
    return result