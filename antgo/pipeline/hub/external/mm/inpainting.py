# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 23:11
# @File    : inpainting.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmedit.apis import init_model, inpainting_inference
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose
import numpy as np


def _inpainting_inference(model, masked_img, mask):
  """Inference image with the model.

  Args:
      model (nn.Module): The loaded model.

  Returns:
      Tensor: The predicted inpainting result.
  """
  device = next(model.parameters()).device  # model device

  infer_pipeline = [
    # dict(type='LoadImageFromFile', key='masked_img'),
    # dict(type='LoadMask', mask_mode='file', mask_config=dict()),
    dict(type='Pad', keys=['masked_img', 'mask'], mode='reflect'),
    dict(
      type='Normalize',
      keys=['masked_img'],
      mean=[127.5] * 3,
      std=[127.5] * 3,
      to_rgb=False),
    dict(type='GetMaskedImage', img_name='masked_img'),
    dict(
      type='Collect',
      keys=['masked_img', 'mask'],
      meta_keys=[]),
    dict(type='ImageToTensor', keys=['masked_img', 'mask'])
  ]

  # build the data pipeline
  test_pipeline = Compose(infer_pipeline)
  # prepare data
  data = dict(masked_img=masked_img, mask=mask)
  data = test_pipeline(data)
  data = collate([data], samples_per_gpu=1)
  if 'cuda' in str(device):
    data = scatter(data, [device])[0]
  else:
    data.pop('meta')
  # forward the model
  with torch.no_grad():
    result = model(test_mode=True, **data)

  return result['fake_img']


class Inpainting(object):
  def __init__(self, config_file, checkpoint_file, device='cpu'):
    self.model = init_model(config_file, checkpoint_file, device=device)

  def __call__(self, *args, **kwargs):
    masked_img = None
    mask = None
    if len(args) == 2:
      masked_img, mask = args
      if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
      else:
        mask = mask[:,:,:1]
    else:
      masked_img = args[0]
      h,w = masked_img.shape[:2]
      mask = np.zeros((h,w,1))

    result = _inpainting_inference(self.model, masked_img, mask)
    result = tensor2img(result, min_max=(-1, 1))[..., ::-1]

    return result['fake_img']