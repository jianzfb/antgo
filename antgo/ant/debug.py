# -*- coding: UTF-8 -*-
# @Time    : 2018/11/9 6:00 PM
# @File    : debug.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.utils import logger
from antgo.context import *
from antgo.dataflow.dataset.random_dataset import *
from antgo.utils.utils import *
from antgo.dataflow.recorder import *
import os
import sys
import yaml
from datetime import datetime
from types import FunctionType
from antgo import config
from antgo.dataflow.recorder import *
from antgo.dataflow.common import *
import traceback

def debug_training_process(dataset_func, param_config=None, dump_dir=None):
  # 0.step get global context
  ctx = get_global_context()
  ctx.debug = True

  # 1.step parse params config file
  if param_config is not None:
    logger.info('load param file %s'% param_config)
    params = yaml.load(open(param_config, 'r'))
    ctx.params = params

  # 2.step call traing process
  if dump_dir is None:
    train_time_stamp = datetime.fromtimestamp(timestamp()).strftime('%Y%m%d.%H%M%S.%f')
    logger.info('build dump folder %s'%train_time_stamp)
    dump_dir = os.path.join(os.curdir, 'dump', train_time_stamp)

  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)
  logger.info('start debug training process')

  dataset_obj = None
  if isinstance(dataset_func, FunctionType):
    dataset_obj = RandomDataset('train', '')
    dataset_obj.data_func = dataset_func
  else:
    Config = config.AntConfig
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)
    dataset_obj = dataset_func('train', os.path.join(Config.data_factory, dataset_func.__name__))

  ctx.call_training_process(dataset_obj, dump_dir=dump_dir)


def debug_infer_process(dataset_func, param_config=None, dump_dir=None):
  # 0.step get global context
  ctx = get_global_context()
  ctx.debug = True

  # 1.step parse params config file
  if param_config is not None:
    logger.info('load param file %s' % param_config)
    params = yaml.load(open(param_config, 'r'))
    ctx.params = params

  # 2.step call traing process
  if dump_dir is None:
    train_time_stamp = datetime.fromtimestamp(timestamp()).strftime('%Y%m%d.%H%M%S.%f')
    logger.info('build dump folder %s'%train_time_stamp)
    dump_dir = os.path.join(os.curdir, 'dump', train_time_stamp)
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)
  logger.info('start debug infer process')
  ctx.recorder = EmptyRecorderNode()
  ctx.ant = True

  dataset_obj = None
  if isinstance(dataset_func, FunctionType):
    dataset_obj = RandomDataset('test', '')
    dataset_obj.data_func = dataset_func
    ctx.call_infer_process(dataset_obj, dump_dir=dump_dir)
  else:
    Config = config.AntConfig
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)
    dataset_obj = dataset_func('test', os.path.join(Config.data_factory, dataset_func.__name__))
    dataset_obj = DataAnnotationBranch(Node.inputs(dataset_obj))
    ctx.recorder = RecorderNode(Node.inputs(dataset_obj.output(1)))

    ctx.call_infer_process(dataset_obj.output(0), dump_dir=dump_dir)
