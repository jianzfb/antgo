# -*- coding: UTF-8 -*-
# @Time    : 17-9-5
# @File    : convert.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .base import *


class AntConvert(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_token,
               ant_task_config):
    super(AntConvert,self).__init__(ant_name, ant_context, ant_token)
  
  def start(self):
    pass