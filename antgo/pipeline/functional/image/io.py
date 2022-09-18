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


@register
def image_decode(x):
  image = cv2.imread(x)
  return image

@register
def image_base64_decode(x):
  content = base64.b64decode(x)
  image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
  return image

@register
class image_save(object):
  def __init__(self, folder):
    self.folder = folder

  def __call__(self, file_path, x):
    file_name = file_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(self.folder, f'{file_name}.png'), x)
