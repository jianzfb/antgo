# encoding=utf-8
# @Time    : 17-6-22
# @File    : crop.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.core import *
from antgo.utils.bboxes import *


class RandomCrop(Node):
  def __init__(self, inputs, crop_size, min_overlap=0.3):
    super(RandomCrop, self).__init__(name=None, action=self.action, inputs=inputs)
    self._crop_size = crop_size
    self._min_overlap = min_overlap
  
  def action(self, *args, **kwargs):
    assert(len(args) == 1)

    data, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    height, width, _ = data.shape