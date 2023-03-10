import sys
from antgo.ant import environment
from antgo.dataflow.datasetio import *
import os
import json
import pickle
from antgo.framework.helper.dataset.tfdataset import *
from antgo.framework.helper.dataset.kvdataset import *
from antgo.ant import environment
import requests
import logging


class __SampleDataGenerator(object):
    def __init__(self, json_file) -> None:
        self.src_folder = os.path.dirname(json_file)
        with open(json_file, 'r') as fp:
            self.samples = json.load(fp)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for sample in self.samples:
            # 转换通用数据格式
            for k in sample.keys():
                if k in ['image_url', 'image_file']:
                    continue
                
                # 对于list数据直接转换为numpy
                if isinstance(sample[k], list):
                    sample[k] = np.array(sample[k])
                    # 转换浮点数值
                    if sample[k].dtype in [np.float64, np.float16]:
                        sample[k] = sample[k].astype(np.float32)
                    # 转换bool数值
                    if sample[k].dtype in [np.bool8]:
                        sample[k] = sample[k].astype(np.int8)

                # 对bool数据转换成int
                if isinstance(sample[k], bool):
                    sample[k] = int(sample[k])

            # 加载图像
            assert(sample['image_file'] != '' or sample['image_url'] != '')
            if sample['image_url'] != '':
                try:
                    pic = requests.get(sample['image_url'], timeout=20)
                    sample['image'] = pic.content                
                except:
                    logging.error("Couldnt download %s."%sample['image_url'])
                    sample['image'] = b''
            else:
                image_path = os.path.join(self.src_folder, sample['image_file'])
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as fp:
                        image_content = fp.read()
                        sample['image'] = image_content
                else:
                    print(f'Missing image {image_path}')
                    sample['image'] = b''
            
            if sample['image'] == b'':
                # 图像为空时，直接忽略样本
                yield None
            else:            
                yield sample


def package_to_kv(src_file, tgt_folder, prefix, size_in_shard=-1, **kwargs):
    # src_file json 文件 (仅支持标准格式 sample_gt.json)

    # 创建writer 实例
    kvw = KVDataWriter(prefix, tgt_folder, -1)

    # 
    kvw.write(__SampleDataGenerator(src_file))


def package_to_tfrecord(src_file, tgt_folder, prefix, size_in_shard=-1, **kwargs):
    # src_file json 文件 (仅支持标准格式 sample_gt.json)

    # 创建tfdatawriter 实例
    if size_in_shard <= 0:
        size_in_shard = 100000
    tfw = TFDataWriter(prefix, tgt_folder, size_in_shard)

    # 写数据
    tfw.write(__SampleDataGenerator(src_file))


# test tfrecord
# package_to_tfrecord("/root/workspace/ss/annotation.json", "/root/workspace/BB", "hello", 10000)

# tfd = TFDataset(data_path_list=['/root/workspace/BB/hello-00000-of-00001-tfrecord'], shuffle_queue_size=2)
# for a in tfd:
#     print(a)

# test kv
# package_to_kv("/root/workspace/ss/annotation.json", "/root/workspace/BB", "hello")
# kvd = KVDatasetReader(data_path_list=['/root/workspace/BB/hello-00000-of-00001-kvrecord'])
# print(len(kvd))

# index = [10,20]
# result = kvd.reads(index)
# print(result)