# -*- coding: UTF-8 -*-
# Time: 12/31/17
# File: publish_dataset.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.context import *
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
  if dataset_flag == 'train':
    a = np.random.random((100))
    b = np.random.randint(0,10, 100)
    for aa, bb in zip(a.flatten(), b.flatten()):
      yield aa,bb
  elif dataset_flag == 'val':
    a = np.random.random((100))
    b = np.random.randint(0,10, 100)
    for aa, bb in zip(a.flatten(), b.flatten()):
      yield aa,bb
  else:
    a = np.random.random((100))
    b = np.random.randint(0,10, 100)
    for aa, bb in zip(a.flatten(), b.flatten()):
      yield aa,bb


###################################################
####### 3.step register generate callback #########
#######                                   #########
###################################################
ctx.data_generator = dataset_callback