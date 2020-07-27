# -*- coding: UTF-8 -*-
# @Time    : 2019-06-06 22:19
# @File    : preparetfrecords.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.dataset.pascal_voc import *
from antgo.dataflow.dataset.dataset import *
import tensorflow as tf

import numpy as np

train_or_test = 'train'
pascal_dataset = Pascal2012(train_or_test,
                            '/Users/jian/Downloads/factory/dataset/VOC2012',
                            {'task_type': 'SEGMENTATION',
                             'aug': True})

# count = 0
# for data, label in pascal_dataset.iterator_value():
#     if 'segmentation_map' in label:
#         print(count)
#         count += 1
#
#         rgb_image = data
#         label_image = label['segmentation_map']
#         print(label_image.shape)
#
#         imwrite('./dataset/'+train_or_test+'/%d.png'%count, rgb_image)
#         imwrite('./dataset/'+train_or_test+'/%d_mask.png'%count, label_image)


def _get_dataset_record_filename(dataset_record_dir, dataset_name, split_name, shard_id, num_shards):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name, split_name, shard_id, num_shards)
  return os.path.join(dataset_record_dir, output_filename)


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

image_list = []
image_mask_list = []
for f in os.listdir('./dataset/'+train_or_test+'/'):
    if f[0] == '.':
        continue

    if '_mask' not in f:
        continue

    image_list.append('./dataset/'+train_or_test+'/' + f.replace('_mask',''))
    image_mask_list.append('./dataset/'+train_or_test+'/' + f)

num_shards = 20
num_per_shard = int(np.math.ceil(len(image_list) / float(num_shards)))
with tf.Session('') as sess:
    for shard_id in range(num_shards):
        record_filename = _get_dataset_record_filename('./dataset/tfrecords', 'voc2012', train_or_test, shard_id, num_shards)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(image_list))

            for i in range(start_ndx, end_ndx):
                print('process i %d\n'%i)
                format_str = 'png'
                image_bin_data = tf.gfile.FastGFile(image_list[i], 'rb').read()
                mask_bin_data = tf.gfile.FastGFile(image_mask_list[i], 'rb').read()
                example = tf.train.Example(features=
                                        tf.train.Features(feature={
                                            'image/encoded': _bytes_feature(image_bin_data),
                                            'image/format': _bytes_feature(format_str.encode('utf-8')),
                                            'mask/encoded': _bytes_feature(mask_bin_data),
                                            'mask/format': _bytes_feature(format_str.encode('utf-8'))}))

                tfrecord_writer.write(example.SerializeToString())

