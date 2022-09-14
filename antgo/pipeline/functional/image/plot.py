# -*- coding: UTF-8 -*-
# @Time    : 2022/9/14 23:25
# @File    : plot.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from antgo.pipeline.engine import *
import cv2

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
  # Plots one bounding box on image img
  tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
  color = color or [random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


@register
class plot_bbox(object):
  def __init__(self, color=None, line_thickness=1):
    self.color = color
    self.line_thickness = line_thickness

  def __call__(self, image, detbbox, detlabel):
    for (*xyxy, conf, cls), label in zip(detbbox, detlabel):
      bbox_label = str(int(cls))
      if label != '':
        bbox_label = label

      bbox_color = [random.randint(0, 255) for _ in range(3)]
      if self.color is not None:
        bbox_color = self.color[int(cls)]
      plot_one_box(xyxy, image, label=bbox_label, color=bbox_color, line_thickness=self.line_thickness)

    return image