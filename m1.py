# -*- coding: UTF-8 -*-
# @Time    : 2022/4/23 23:10
# @File    : m1.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.reader import *
from antgo.dataflow.dataset.mnist import *
from antgo.dataflow.imgaug.operators import *

data_folder = '/Users/jian/Downloads/factory/dataset/MNIST'
dataset = Mnist('train', data_folder)
reader = Reader(dataset, [
  RandomFlipImage(),
  RandomDistort(),
  AutoAugmentImage()
], None, 4, True, True, inputs_def={
  'fields':{'image': {},
  'id': {}}
})

for data in reader.iterator_value():
  print(data)