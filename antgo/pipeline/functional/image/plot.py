# -*- coding: UTF-8 -*-
# @Time    : 2022/9/14 23:25
# @File    : plot.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import random
from antgo.pipeline.engine import *
from antgo.utils import colormap
import cv2
import numpy as np


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
  def __init__(self, thres=0.0, color=None, line_thickness=1, category_map=None, ignore_category=-1):
    self.color = color
    self.line_thickness = line_thickness
    self.thres = thres
    self.category_map = category_map
    self.ignore_category = ignore_category

  def __call__(self, image, detbbox, detlabel=None):
    if detbbox.shape[0] == 0:
      return image

    if detlabel is not None:
      if detlabel.shape[0] == 0:
        return image
      if len(detlabel.shape) == 1:
        detlabel = detlabel[:,None]
      detbbox = np.concatenate([detbbox, detlabel], -1)

    # format 1: x0,y0,x1,y1,c
    # format 2: x0,y0,x1,y1,score,c
    assert(detbbox.shape[1] == 5 or detbbox.shape[1] == 6)

    if detbbox.shape[1] == 6:
      for (*xyxy, conf, class_id) in detbbox:
        if self.ignore_category >= 0:
          if int(class_id) == self.ignore_category:
            continue
    
        bbox_label = str(int(class_id))
        if self.category_map is not None:
          bbox_label = self.category_map[bbox_label]

        bbox_color = [random.randint(0, 255) for _ in range(3)]
        if self.color is not None:
          bbox_color = self.color[int(bbox_label)]
        plot_one_box(xyxy, image, label=bbox_label, color=bbox_color, line_thickness=self.line_thickness)
    else:
      for (*xyxy, class_id) in detbbox:
        if self.ignore_category >= 0:
          if int(class_id) == self.ignore_category:
            continue
  
        bbox_label = str(int(class_id))
        if self.category_map is not None:
          bbox_label = self.category_map[bbox_label]
        
        bbox_color = [random.randint(0, 255) for _ in range(3)]
        if self.color is not None:
          bbox_color = self.color[int(bbox_label)]
        plot_one_box(xyxy, image, label=bbox_label, color=bbox_color, line_thickness=self.line_thickness)

    return image
  

@register
class plot_text(object):
  def __init__(self, label_map=None, color=None, pos=None, font_scale=1, thickness=1):
    self.pos = pos
    self.color = color
    self.thickness = thickness
    self.font_scale = font_scale
    self.label_map = label_map
  
  def __call__(self, image, text):
    pos = self.pos
    if pos is None:
      pos = (int(image.shape[1]/2), int(image.shape[0]/2))
    
    color = self.color
    if color is None:
      color = (0, 0, 255)
    
    if not isinstance(text, str):
      text = self.label_map[int(text)]
    image = cv2.putText(image, text, pos, cv2.FONT_HERSHEY_COMPLEX, self.font_scale, color, self.thickness)
    return image


@register
class plot_poly(object):
  def __init__(self, fill=1, is_overlap=False):
    self.fill = fill
    self.is_overlap = is_overlap
  
  def __call__(self, image, polygon_points):
    if self.is_overlap:
      image_cp = image.copy()
      cv2.fillPoly(image_cp, polygon_points, self.fill)
      return image_cp
    else:
      mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
      mask = cv2.fillPoly(mask, np.array(polygon_points).astype(np.int32), self.fill)
      image_with_mask = np.concatenate([image, np.expand_dims(mask, -1)], -1)
      return image_with_mask
