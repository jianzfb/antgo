# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile
slim = tf.contrib.slim
from antgo.dataflow.dataset import *
import scipy.misc


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _get_dataset_record_filepath(dataset_record_dir, dataset_name, split_name, shard_id, num_shards):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name, split_name, shard_id, num_shards)
  return os.path.join(dataset_record_dir, output_filename)


class AntTFRecords(object):
    def __init__(self, ant_context, data_factory, dataset, dump_dir):
        self.num_shards = 100
        self.dump_dir = dump_dir
        self.ant_context = ant_context
        self.data_factory = data_factory
        self.dataset = dataset

    def __make_records(self, data):
        reorganized_data = {}
        for key, value in data.items():
            xxyy = key.split('_')
            data_name = xxyy[0]
            if data_name not in reorganized_data:
                reorganized_data[data_name] = {}

            if len(xxyy) == 1:
                reorganized_data[data_name]['data'] = value
            elif xxyy[1] == 'TYPE':
                reorganized_data[data_name]['type'] = value

        tf_data = {}
        for key, value in reorganized_data.items():
            v_data = value['data']
            v_type = value['type']
            if v_type == "IMAGE":
                # 保存临时文件，使用tf.gfile.FastGFile
                scipy.misc.imsave("./temp.png", v_data)
                v_data = tf.gfile.FastGFile('./temp.png', 'rb').read()
                tf_data[key] = _bytes_feature(v_data)
            elif v_type == "STRING":
                tf_data[key] = _bytes_feature(v_data.encode('utf-8'))
            elif v_type == "INT":
                tf_data[key] = _int64_feature(v_data)

        return tf_data

    def start(self):
        parse_flag = ''
        dataset_cls = AntDatasetFactory.dataset(self.dataset, parse_flag)

        if dataset_cls is None:
            logger.error('couldnt find dataset parse class')
            return

        # 4.step 启动数据生成
        # 4.1.step 训练集
        train_dataset = dataset_cls('train', os.path.join(self.data_factory, self.dataset))

        if train_dataset.size > 0:
            # 确保每个shard中至少100个样本
            num_shards = self.num_shards
            num_per_shard = int(np.math.ceil(train_dataset.size / float(num_shards)))
            if num_per_shard < 200:
                num_shards = int(np.math.ceil(train_dataset.size / 200))
                num_per_shard = int(np.math.ceil(train_dataset.size / num_shards))

            logger.info('train dataset -> tfrecords (%d shards, %d samples in a shard)'%(num_shards, num_per_shard))
            sample_index = 0
            try:
                tfrecord_writer = None
                with tf.Session() as sess:
                    record_dir = os.path.join(self.dump_dir, 'train')
                    if not os.path.exists(record_dir):
                        os.makedirs(record_dir)

                    for data in self.ant_context.data_generator(train_dataset):
                        if sample_index % num_per_shard == 0:
                            if tfrecord_writer is not None:
                                tfrecord_writer.close()
                                tfrecord_writer = None

                            record_filepath = \
                                _get_dataset_record_filepath(record_dir,
                                                             self.dataset,
                                                             'train',
                                                             sample_index//num_per_shard,
                                                             num_shards)

                            tfrecord_writer = tf.python_io.TFRecordWriter(record_filepath)
                            logger.info('generate train shard %d'%(sample_index//num_per_shard))

                        sample_index += 1

                        # tf features
                        feature = self.__make_records(data)
                        # warp to tf example
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # write to file
                        tfrecord_writer.write(example.SerializeToString())
            except:
                if tfrecord_writer is not None:
                    tfrecord_writer.close()
                    tfrecord_writer = None

                logger.info('finish training %d samples'%sample_index)

        # 4.2.step 验证集
        val_dataset = dataset_cls('val', os.path.join(self.data_factory, self.dataset))

        if val_dataset.size > 0:
            # 确保每个shard中至少100个样本
            num_shards = self.num_shards
            num_per_shard = int(np.math.ceil(val_dataset.size / float(num_shards)))
            if num_per_shard < 200:
                num_shards = int(np.math.ceil(val_dataset.size / 200))
                num_per_shard = int(np.math.ceil(val_dataset.size / num_shards))

            logger.info('val dataset -> tfrecords (%d shards, %d samples in a shard)'%(num_shards, num_per_shard))
            sample_index = 0
            try:
                tfrecord_writer = None
                with tf.Session() as sess:
                    record_dir = os.path.join(self.dump_dir, 'val')
                    if not os.path.exists(record_dir):
                        os.makedirs(record_dir)

                    for data in self.ant_context.data_generator(val_dataset):
                        if sample_index % num_per_shard == 0:
                            if tfrecord_writer is not None:
                                tfrecord_writer.close()
                                tfrecord_writer = None

                            record_filepath = \
                                _get_dataset_record_filepath(record_dir,
                                                             self.dataset,
                                                             'val',
                                                             sample_index//num_per_shard,
                                                             num_shards)

                            tfrecord_writer = tf.python_io.TFRecordWriter(record_filepath)
                            logger.info('generate val shard %d' % (sample_index // num_per_shard))

                        sample_index += 1

                        # tf features
                        feature = self.__make_records(data)
                        # warp to tf example
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # write to file
                        tfrecord_writer.write(example.SerializeToString())
            except:
                if tfrecord_writer is not None:
                    tfrecord_writer.close()
                    tfrecord_writer = None
                logger.info('finish val %d samples' % sample_index)

        # 4.3.step 测试集
        test_dataset = dataset_cls('test', os.path.join(self.data_factory, self.dataset))

        if test_dataset.size > 0:
            # 确保每个shard中至少100个样本
            num_shards = self.num_shards
            num_per_shard = int(np.math.ceil(test_dataset.size / float(num_shards)))
            if num_per_shard < 200:
                num_shards = int(np.math.ceil(test_dataset.size / 200))
                num_per_shard = int(np.math.ceil(test_dataset.size / num_shards))

            logger.info('test dataset -> tfrecords (%d shards, %d samples in a shard)' % (num_shards, num_per_shard))
            sample_index = 0
            try:
                tfrecord_writer = None
                with tf.Session() as sess:
                    record_dir = os.path.join(self.dump_dir, 'test')
                    if not os.path.exists(record_dir):
                        os.makedirs(record_dir)

                    for data in self.ant_context.data_generator(test_dataset):
                        if sample_index % num_per_shard == 0:
                            if tfrecord_writer is not None:
                                tfrecord_writer.close()
                                tfrecord_writer = None

                            record_filepath = \
                                _get_dataset_record_filepath(record_dir,
                                                             self.dataset,
                                                             'teset',
                                                             sample_index // num_per_shard,
                                                             num_shards)

                            tfrecord_writer = tf.python_io.TFRecordWriter(record_filepath)
                            logger.info('generate test shard %d' % (sample_index // num_per_shard))

                        sample_index += 1

                        # tf features
                        feature = self.__make_records(data)
                        # warp to tf example
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        # write to file
                        tfrecord_writer.write(example.SerializeToString())
            except:
                if tfrecord_writer is not None:
                    tfrecord_writer.close()
                    tfrecord_writer = None

                logger.info('finish val %d samples' % sample_index)
