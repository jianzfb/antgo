# -*- coding: UTF-8 -*-
# @Time    : 2019-02-12 10:22
# @File    : dataset_test.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import unittest
from antgo.dataflow.dataset.coco2017 import *
from antgo import config
from antgo.utils import logger
import os


class TestDataset(unittest.TestCase):
  def test_coco2017(self):
    Config = config.AntConfig
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)

    logger.info('test coco2017 train for SEGMENTATION task (stuff)')
    coco2017 = COCO2017('train', os.path.join(Config.data_factory, 'COCO2017'),
                        {'task_type': 'SEGMENTATION',
                         'task_type_subset': 'stuff'})

    file_num = 0
    for data, annotation in coco2017.iterator_value():
      file_num += 1


    print(file_num)

    logger.info('test coco2017 val for SEGMENTATION task (stuff)')
    coco2017 = COCO2017('val', os.path.join(Config.data_factory, 'COCO2017'),
                        {'task_type': 'SEGMENTATION',
                         'task_type_subset': 'stuff'})
    file_num = 0
    for data, annotation in coco2017.iterator_value():
      file_num += 1

    print(file_num)

    logger.info('test coco2017 test for SEGMENTATION task (stuff)')
    coco2017 = COCO2017('test', os.path.join(Config.data_factory, 'COCO2017'),
                        {'task_type': 'SEGMENTATION',
                         'task_type_subset': 'stuff'})
    file_num = 0
    for data, annotation in coco2017.iterator_value():
      file_num += 1

    print(file_num)

    # coco2017 = COCO2017('train', os.path.join(Config.data_factory, 'COCO2017'),
    #                     {'task_type': 'SEGMENTATION',
    #                      'task_type_subset': 'panoptic'})
    #
    # file_num = 0
    # for data, annotation in coco2017.iterator_value():
    #   file_num += 1
    #
    # print(file_num)

