# encoding=utf-8
# Time: 8/15/17
# File: generate.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import plyvel
from antgo.dataflow.basic import *

def generate_standard_dataset(data_label_generator, data_folder, dataset_name, extra_attrs={}):
  # build db
  if not os.path.exists(os.path.join(data_folder, dataset_name)):
    os.makedirs(os.path.join(data_folder, dataset_name))
  dataset_record = RecordWriter(os.path.join(data_folder, dataset_name))

  # write data and label
  for data, label in data_label_generator:
    dataset_record.write(Sample(data=data, label=label))

  # bind attributes
  if len(extra_attrs) > 0:
    dataset_record.bind_attrs()

  # close dataset
  dataset_record.close()