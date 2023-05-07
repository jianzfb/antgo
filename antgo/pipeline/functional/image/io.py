# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : io.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
import cv2
import os
import base64
import numpy as np
import requests
from antgo.utils.sample_gt import *
import imagesize


@register
class image_decode(object):
  def __init__(self, to_rgb=False) -> None:
    self.to_rgb = to_rgb
  
  def __call__(self, x):
    image = cv2.imread(x)
    if self.to_rgb:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    return image


@register
class image_decode_from_buffer(object):
  def __init__(self, to_rgb=False) -> None:
    self.to_rgb = to_rgb
  
  def __call__(self, x):
    data = np.frombuffer(x, dtype='uint8')
    image = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    if self.to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
    return image  
     

@register
def image_base64_decode(x):
  content = base64.b64decode(x)
  image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
  return image


@register
def image_download(image_url):
  pic = requests.get(image_url, timeout=20)
  image = cv2.imdecode(np.frombuffer(pic.content, np.uint8), cv2.IMREAD_COLOR)  
  return image
  
@register
def serialize_numpy(*args):
  serialize_result = []
  for v in args:
    if isinstance(v, np.ndarray):
      serialize_result.append(v.tolist())
    else:
      serialize_result.append(v)

  return tuple(serialize_result)

@register
class image_save(object):
  def __init__(self, folder, to_bgr=False):
    self.folder = folder
    self.to_bgr = to_bgr
    if not os.path.exists(self.folder):
      os.makedirs(self.folder)

    self.count = 0
    
  def __call__(self, x, file_path=None):
    if self.to_bgr:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    

    file_name = ''
    if file_path is not None:
      file_name = file_path.split('/')[-1].split('.')[0]
    else:
      file_name = f'frame_{self.count}'
      self.count += 1
    cv2.imwrite(os.path.join(self.folder, f'{file_name}.png'), x)


@register
class convert_gt_annos(object):
  def __init__(self, fields, replace_prefix, fill_lambda_fields=None, fill_constant_fields=None) -> None:
    self.fields = fields
    self.replace_prefix = replace_prefix
    self.fill_constant_fields = fill_constant_fields
    self.fill_lambda_fields =  fill_lambda_fields
    
  def __call__(self, *args):
    sgt = SampleGTTemplate()
    gt_info = sgt.get()
    for field, data in zip(self.fields, args):
      if field == 'image':
        gt_info['height'] = data.shape[0]
        gt_info['width'] = data.shape[1]
        continue
      
      if self.fill_lambda_fields is not None:
        if field in self.fill_lambda_fields:
          fill_field = self.fill_lambda_fields[field]['field']
          lambda_func = self.fill_lambda_fields[field]['func']
          gt_info[fill_field] = lambda_func(data)
          continue

      if field not in gt_info:
        continue
              
      if isinstance(gt_info[field], list):
        gt_info[field] = data.tolist()
      elif field in ['image_file', 'semantic_file']:
        if field == 'image_file':
          width, height = imagesize.get(data)
          gt_info['height'] = height
          gt_info['width'] = width
        
        gt_info[field] = data.replace(self.replace_prefix[0], self.replace_prefix[1])
        
      else:
        gt_info[field] = data
    
    if self.fill_constant_fields is not None:
      for k, v in self.fill_constant_fields.items():
        gt_info[k] = v
    return gt_info