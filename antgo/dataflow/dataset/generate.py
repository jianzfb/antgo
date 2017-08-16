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
from antgo.dataflow.dataset.pascal_voc import *


def generate_standard_dataset(data_label_generator, train_or_test, data_folder, dataset_name, extra_attrs={}):
  # build db
  if not os.path.exists(os.path.join(data_folder, dataset_name, train_or_test)):
    os.makedirs(os.path.join(data_folder, dataset_name, train_or_test))
  dataset_record = RecordWriter(os.path.join(data_folder, dataset_name, train_or_test))

  # write data and label
  for data, label in data_label_generator:
    dataset_record.write(Sample(data=data, label=label))

  # bind attributes
  if len(extra_attrs) > 0:
    dataset_record.bind_attrs()

  # close dataset
  dataset_record.close()


def generate_voc2007_standard_dataset(data_folder, target_folder):
  # train dataset
  pascal_train_2007 = Pascal2007('train', data_folder)
  generate_standard_dataset(pascal_train_2007.iterator_value(), 'train', target_folder, 'voc2007')

  # val dataset
  pascal_val_2007 = Pascal2007('val', data_folder)
  generate_standard_dataset(pascal_val_2007.iterator_value(), 'val', target_folder, 'voc2007')


if __name__ == '__main__':
  # transfer voc2007
  #generate_voc2007_standard_dataset('/home/mi/下载/dataset/voc','/home/mi/antgo/antgo-dataset')
  pass
