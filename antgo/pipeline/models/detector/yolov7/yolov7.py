# -*- coding: UTF-8 -*-
# @Time    : 2022/9/13 16:10
# @File    : yolov7.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import yaml

from antgo.pipeline.models.utils.utils import *
from antgo.pipeline.models.utils.preprocess import *
from .util import *
from ..util import *


class Model(nn.Module):
  def __init__(self, model_name='yolov7', device='cpu'):
    super().__init__()
    self.device = device
    model_path = torch_model_download(model_name, file_type='.pth')

    pure_model_name = model_name.split('.')[0]
    model_cfg = \
      os.path.realpath(__file__).split('/')[:-3] + ['cfg', 'detector', f'{pure_model_name}.yaml']
    model_cfg_file = '/'.join(model_cfg)
    assert(os.path.exists(model_cfg_file))
    with open(model_cfg_file) as f:
      self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    # 解析模型，并加载权重
    self.model, self.save = yolo_parse_model(self.yaml, ch=[3])
    self.fuse()
    self.to(self.device)
    self.load_state_dict(torch.load(model_path, map_location=self.device))

    # Build strides, anchors
    m = self.model[-1]  # Detect()
    m.stride = torch.tensor([8,16,32])
    m.anchors /= m.stride.view(-1, 1, 1)
    self.stride = m.stride

  def forward(self, x):
    y = []
    for m in self.model:
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

      x = m(x)  # run

      y.append(x if m.i in self.save else None)  # save output
    return x

  def fuse(self):
    for m in self.model.modules():
      if isinstance(m, RepConv):
        m.fuse_repvgg_block()
      elif isinstance(m, RepConv_OREPA):
        m.switch_to_deploy()
      elif type(m) is Conv and hasattr(m, 'bn'):
        m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
        delattr(m, 'bn')  # remove batchnorm
        m.forward = m.fuseforward  # update forward
    return self


class Yolov7(object):
  def __init__(self, model_name='yolov7', device='cpu', conf_thres=0.25, iou_thres=0.45, dataset='coco'):
    self.device = select_device(device)
    self.model = Model(f'{model_name}.{dataset.lower()}', self.device)
    self.conf_thres = conf_thres
    self.iou_thres = iou_thres

    dataset_cfg = \
      os.path.realpath(__file__).split('/')[:-3] + ['cfg', 'dataset', f'{dataset.lower()}.yaml']
    dataset_cfg_file = '/'.join(dataset_cfg)
    with open(dataset_cfg_file) as f:
      self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    self.names = self.yaml['names']

  def __call__(self, *args, **kwargs):
    img0 = args[0]
    img0_shape = img0.shape
    # Padded resize
    img = paddedResize(img0, (640, 640), stride=32)[0]
    img_shape = img.shape

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(self.device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)

    pred = self.model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, agnostic=True)

    if len(pred) == 0:
      return np.empty((0,5), dtype=np.float32), []

    det = pred[0]
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img_shape[:2], det[:, :4], img0_shape[:2]).round()

    det_names = []
    for *xyxy, conf, cls in reversed(det):
      cls_name = self.names[(int)(cls)]
      det_names.append(cls_name)

    det = det.detach().cpu().numpy()
    return det, det_names
