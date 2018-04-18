# -*- coding: UTF-8 -*-
# @Time    : 18-4-18
# @File    : template.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import shutil
import os
from antgo.utils import logger
import traceback

class AntTemplate(object):
  def __init__(self, *args, **kwargs):
    self.dump_dir = kwargs['dump_dir']
    self.main_file = kwargs['main_file']
    self.main_param = kwargs['main_param']
  
  def start(self):
    try:
      # 1.step get template resource folder
      file_folder = os.path.dirname(__file__)
      parent_folder = '/'.join(file_folder.split('/')[:-1])
      template_file_folder = os.path.join(parent_folder, 'resource', 'templates')
      
      # 2.step copy main_file.py
      main_file = 'task_main_file.py' if self.main_file is None else self.main_file
      shutil.copy(os.path.join(template_file_folder, 'task_main_file.py'), os.path.join(self.dump_dir, main_file))
  
      # 3.step copy main_param.yaml
      main_param = 'task_main_param.yaml' if self.main_param is None else self.main_param
      shutil.copy(os.path.join(template_file_folder, 'task_main_param.yaml'), os.path.join(self.dump_dir, main_param))
      
      logger.info('execute template command')
    except:
      logger.error('fail execute template command')
      traceback.print_exc()
      