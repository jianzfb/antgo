# -*- coding: UTF-8 -*-
# Time: 12/31/17
# File: publish_challenge_dataset.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.context import *
from antgo.measures.base import *
from antgo.measures import *
from antgo.dataflow.dataset import *
import numpy as np
##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


###################################################
####### 2.step custom data generator      #########
#######                                   #########
###################################################
def dataset_callback(dataset_flag):
  # if dataset_flag == 'train':
  #   a = np.random.random((ctx.params.train['num']))
  #   b = np.random.randint(0,10, ctx.params.train['num'])
  #   for aa, bb in zip(a.flatten(), b.flatten()):
  #     yield aa,bb
  # elif dataset_flag == 'val':
  #   a = np.random.random((ctx.params.val['num']))
  #   b = np.random.randint(0,10, ctx.params.val['num'])
  #   for aa, bb in zip(a.flatten(), b.flatten()):
  #     yield aa,bb
  # else:
  #   a = np.random.random((ctx.params.test['num']))
  #   b = np.random.randint(0,10, ctx.params.test['num'])
  #   for aa, bb in zip(a.flatten(), b.flatten()):
  #     yield aa,bb
  if dataset_flag == 'train':
    for _ in range(ctx.params.train['num']):
      image = np.random.random((100,100))
      mask = np.random.randint(0,1,(100,100))
      yield  image, mask
  elif dataset_flag == 'test':
    for _ in range(ctx.params.test['num']):
      image = np.random.random((100,100))
      mask = np.random.randint(0,1,(100,100))
      yield image, mask


###################################################
####### 3.step register generate callback #########
#######                                   #########
###################################################
ctx.data_generator = dataset_callback