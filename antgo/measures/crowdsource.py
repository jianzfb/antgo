# -*- coding: UTF-8 -*-
# Time: 1/2/18
# File: crowdsource.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.task.task import *
from antgo.measures.base import *

class Crowdsource(AntMeasure):
  def __init__(self, task, name):
    super(Crowdsource, self).__init__(task, name)


  def eva(self, data, label):
    # 1.step http server

    # 2.step
    pass