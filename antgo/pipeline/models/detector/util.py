# -*- coding: UTF-8 -*-
# @Time    : 2022/9/13 13:17
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml


def xyxy2xywh(x):
  # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
  y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
  y[:, 2] = x[:, 2] - x[:, 0]  # width
  y[:, 3] = x[:, 3] - x[:, 1]  # height
  return y


def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
  # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
  y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
  y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
  y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
  return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
  # Convert normalized segments into pixel segments, shape (n,2)
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = w * x[:, 0] + padw  # top left x
  y[:, 1] = h * x[:, 1] + padh  # top left y
  return y


def segment2box(segment, width=640, height=640):
  # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
  x, y = segment.T  # segment xy
  inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
  x, y, = x[inside], y[inside]
  return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
  # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
  boxes = []
  for s in segments:
    x, y = s.T  # segment xy
    boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
  return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
  # Up-sample an (n,2) segment
  for i, s in enumerate(segments):
    s = np.concatenate((s, s[0:1, :]), axis=0)
    x = np.linspace(0, len(s) - 1, n)
    xp = np.arange(len(s))
    segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
  return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
  # Rescale coords (xyxy) from img1_shape to img0_shape
  if ratio_pad is None:  # calculate from img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
  else:
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]

  coords[:, [0, 2]] -= pad[0]  # x padding
  coords[:, [1, 3]] -= pad[1]  # y padding
  coords[:, :4] /= gain
  clip_coords(coords, img0_shape)
  return coords


def clip_coords(boxes, img_shape):
  # Clip bounding xyxy bounding boxes to image shape (height, width)
  boxes[:, 0].clamp_(0, img_shape[1])  # x1
  boxes[:, 1].clamp_(0, img_shape[0])  # y1
  boxes[:, 2].clamp_(0, img_shape[1])  # x2
  boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
  # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
  box2 = box2.T

  # Get the coordinates of bounding boxes
  if x1y1x2y2:  # x1, y1, x2, y2 = box1
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
  else:  # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

  # Intersection area
  inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
          (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

  # Union Area
  w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
  w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
  union = w1 * h1 + w2 * h2 - inter + eps

  iou = inter / union

  if GIoU or DIoU or CIoU:
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
      c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
      rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
              (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
      if DIoU:
        return iou - rho2 / c2  # DIoU
      elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
          alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
    else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
      c_area = cw * ch + eps  # convex area
      return iou - (c_area - union) / c_area  # GIoU
  else:
    return iou  # IoU


def fuse_conv_and_bn(conv, bn):
  # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
  fusedconv = nn.Conv2d(conv.in_channels,
                        conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        groups=conv.groups,
                        bias=True).requires_grad_(False).to(conv.weight.device)

  # prepare filters
  w_conv = conv.weight.clone().view(conv.out_channels, -1)
  w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
  fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

  # prepare spatial bias
  b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
  b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
  fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

  return fusedconv


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
  """Runs Non-Maximum Suppression (NMS) on inference results

  Returns:
       list of detections, on (n,6) tensor per image [xyxy, conf, cls]
  """

  nc = prediction.shape[2] - 5  # number of classes
  xc = prediction[..., 4] > conf_thres  # candidates

  # Settings
  min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
  max_det = 300  # maximum number of detections per image
  max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
  time_limit = 10.0  # seconds to quit after
  redundant = True  # require redundant detections
  multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
  merge = False  # use merge-NMS

  t = time.time()
  output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
  for xi, x in enumerate(prediction):  # image index, image inference
    # Apply constraints
    # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    x = x[xc[xi]]  # confidence

    # Cat apriori labels if autolabelling
    if labels and len(labels[xi]):
      l = labels[xi]
      v = torch.zeros((len(l), nc + 5), device=x.device)
      v[:, :4] = l[:, 1:5]  # box
      v[:, 4] = 1.0  # conf
      v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
      x = torch.cat((x, v), 0)

    # If none remain process next image
    if not x.shape[0]:
      continue

    # Compute conf
    if nc == 1:
      x[:, 5:] = x[:, 4:5]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
      # so there is no need to multiplicate.
    else:
      x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
      i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
      x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
      conf, j = x[:, 5:].max(1, keepdim=True)
      x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
      x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Apply finite constraint
    # if not torch.isfinite(x).all():
    #     x = x[torch.isfinite(x).all(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
      continue
    elif n > max_nms:  # excess boxes
      x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
      i = i[:max_det]
    if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
      # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
      iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
      weights = iou * scores[None]  # box weights
      x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
      if redundant:
        i = i[iou.sum(1) > 1]  # require redundancy

    output[xi] = x[i]
    if (time.time() - t) > time_limit:
      print(f'WARNING: NMS time limit {time_limit}s exceeded')
      break  # time limit exceeded

  return output

