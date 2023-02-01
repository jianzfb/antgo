from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
import numpy as np
from typing import List
import pickle
import os
from functools import partial
import cv2
from antgo.ant import environment
from antgo.dataflow.dataset import *


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _decode_int(tensor):
    if isinstance(tensor, list):
        return tensor[0]
    return tensor.numpy().item()


def _decode_float(tensor):
    if isinstance(tensor, list):
        return tensor[0]
    return tensor.numpy().item()


def _decode_bytes(tensor):
    # tensor = io.BytesIO(tensor)
    if isinstance(tensor, bytes):
        return np.frombuffer(tensor, dtype=np.uint8)
    return np.frombuffer(tensor.numpy(), dtype=np.uint8)

def _decode_image(tensor, color=False):
    buffer = _decode_bytes(tensor)
    method = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(buffer, method)


def _decode_numpy_array(tensor, dtype=np.float32):
    buffer = _decode_bytes(tensor)
    # return np.frombuffer(buffer, dtype=dtype)
    return pickle.loads(buffer)


def _decode_str(tensor):
    if isinstance(tensor, bytes):
        return tensor.decode("utf-8")
    return tensor.numpy().decode("utf-8")


def _decode_with_pickle(tensor):
    # for list and dict ...
    buffer = _decode_bytes(tensor)
    return pickle.loads(buffer)


class TFRecordData(object):
    def __init__(self, fields2types, path, use_color_image=False) -> None:
        tfrecords_files = []
        for file in os.listdir(path):
            if file.startswith("tfrecords"):
                tfrecords_files.append(os.path.join(path, file))
        
        self.tfrecord_dataset = tf.data.TFRecordDataset(tfrecords_files)
        self.fields2types = fields2types
        self.decode_methods = {}
        for field, ftype in self.fields2types.items():
            if field == "image":
                self.decode_methods[field] = partial(_decode_image, color=use_color_image)
            elif ftype == np.ndarray:
                self.decode_methods[field] = _decode_numpy_array
            elif ftype == bytes:
                self.decode_methods[field] = _decode_bytes
            elif ftype == int:
                self.decode_methods[field] = _decode_int
            elif ftype == float:
                self.decode_methods[field] = _decode_float
            elif ftype == str:
                self.decode_methods[field] = _decode_str
            else:
                self.decode_methods[field] = _decode_with_pickle

    def get_features(self):
        feature = {}
        for field, ftype in self.fields2types.items():
            if ftype == bytes or ftype == str:
                feature[field] = tf.io.FixedLenFeature([], tf.string, default_value="")
            elif ftype == int:
                feature[field] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
            elif ftype == float:
                feature[field] = tf.io.FixedLenFeature([], tf.float32, default_value=0)
            else:
                feature[field] = tf.io.FixedLenFeature([], tf.string, default_value="")

        return feature

    @property
    def size(self):
        return len(self.tfrecord_dataset)

    def parse(self, data):
        decode_data = {}
        for key in data.keys():
            decode_data[key] = self.decode_methods[key](data[key])
        
        return decode_data


class TFRecordDataWriter(object):
    def __init__(self, fields2types, output, num_shards=1) -> None:
        # fields2types = {
        #     "image": bytes,  # 原始图片
        #     "height": int,
        #     "width": int,
        #     "num_joints": int,  # joints数量
        #     # 关键点，包括2d和3d
        #     "joints_2d": np.ndarray,
        #     "joints_3d": np.ndarray,
        #     "joints_25d": np.ndarray,
        #     # 关键点是否可见，包括2d和3d
        #     "joints_vis_2d": np.ndarray,
        #     "joints_vis_3d": np.ndarray,
        #     # bbox坐标
        #     "bbox": list,
        #     # root id
        #     "root_id": int,
        #     # 骨长关键点id，用于normalization
        #     "bone": list,
        #     "bone_scale": float,
        #     "pose": np.ndarray,
        #     "shape": np.ndarray,
        #     "root_3d": np.ndarray,
        #     "bones_template": np.ndarray,
        #     "depth": np.ndarray,
        #     "extra_fields": dict,
        #     # 部分数据集没有深度信息
        #     "has_depth": int,
        # }
        self.fields2types = fields2types
        self.output = output
        self.num_shards = num_shards

        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def get_example_object(self, data_record):
        # 将数据转化为int64 float 或bytes类型的列表
        # 注意都是list形式

        feature_key_value_pair = {
            'int_list':tf.train.Feature(int64_list = tf.train.Int64List(value = [data_record['int_data']])),
            'float_list': tf.train.Feature(float_list=tf.train.FloatList(value = [data_record['float_data']])),
            'str_list': tf.train.Feature(bytes_list=tf.train.BytesList(value = [data_record['str_data']])),
            'float_list2': tf.train.Feature(float_list=tf.train.FloatList(value = data_record['float_list_data'])),
        }
        
        # 创建一个features
        features = tf.train.Features(feature = feature_key_value_pair)
        
        # 创建一个example
        example = tf.train.Example(features = features)
        
        return example
    
    def serialize_field(self, data):
        if isinstance(data, (bytes, str)):
            return _bytes_feature(data)
        if isinstance(data, int):
            return _int64_feature(data)
        if isinstance(data, float):
            return _float_feature(data)
        if isinstance(data, (list, dict, np.ndarray)):
            return _bytes_feature(pickle.dumps(data))
        else:
            raise ValueError("Unknown type: {}".format(type(data)))

    def write(self, data_iterator):
        fields = self.fields2types.keys()
        data_num = len(data_iterator)
        size_in_shard = (data_num+self.num_shards-1) // self.num_shards

        tfwriter = None
        shard_i = -1
        count = 0
        for data in data_iterator:
            features = {}
            for field in fields:
                value = data[field]
                features[field] = self.serialize_field(value) 
            
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            
            if (count // size_in_shard) != shard_i:
                shard_i = count // self.num_shards
                filename = os.path.join(self.output, "tfrecords-%.5d-of-%.5d"%(shard_i, self.num_shards))
                tfwriter = tf.compat.v1.python_io.TFRecordWriter(filename)
            
            tfwriter.write(example_proto.SerializeToString())
            count += 1



