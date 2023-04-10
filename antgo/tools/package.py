import sys
from antgo.dataflow.datasetio import *
import os
import json
import pickle
from antgo.framework.helper.dataset.tfdataset import *
from antgo.framework.helper.dataset.kvdataset import *
from antgo.ant import environment
import requests
import logging
from queue import Queue
import threading
import time


class __SampleDataGenerator(object):
    def __init__(self, json_files, thread_num=1) -> None:
        self.json_file_list = json_files.split(',')

        # 预先加载，获得样本总数
        self.num_sample = 0
        self.num_sample_offset = [0]
        for json_file in self.json_file_list:
            with open(json_file, 'r') as fp:
                content = json.load(fp)
                self.num_sample += len(content)
                self.num_sample_offset.append(self.num_sample_offset[-1]+len(content))
        assign_sample_num = (self.num_sample + (thread_num - 1)) // thread_num

        self.thread_stop_flag = [
            1 for _ in range(thread_num)
        ]
        # 启动线程池进行数据生产
        # 数据队列 (队列最大容量10000)
        self.data_cache = Queue(10000)        
        self.thread_pool = [
            threading.Thread(target=self.produce, args=(i*assign_sample_num, min((i+1)*assign_sample_num, self.num_sample), i, self.data_cache)) for i in range(thread_num)]

        for thread_i in range(thread_num):
            self.thread_pool[thread_i].start()

    def produce(self, start_i, stop_i, thread_id, data_cache):
        # 发现从start_i 到 stop_i，来自于的json_file集合
        start_file_i = 0
        start_sample_offset = 0
        stop_file_i = 0
        stop_sample_offset = 0
        for file_i in range(len(self.json_file_list)):
            if start_i >= self.num_sample_offset[file_i] and start_i <= self.num_sample_offset[file_i+1]:
                start_file_i = file_i
                start_sample_offset = start_i - self.num_sample_offset[file_i]

            if stop_i >= self.num_sample_offset[file_i] and stop_i <= self.num_sample_offset[file_i+1]:
                stop_file_i = file_i
                stop_sample_offset = stop_i - self.num_sample_offset[file_i]

        for file_i in range(start_file_i, stop_file_i+1):
            json_file = self.json_file_list[file_i]
            src_folder = os.path.dirname(json_file)
            logging.info(f'Process {src_folder}')
            with open(json_file, 'r', encoding="utf-8") as fp:
                samples = json.load(fp)

            start_ii = 0
            stop_ii = len(samples)
            if file_i == start_file_i:
                start_ii = start_sample_offset

            if file_i == stop_file_i:
                stop_ii = min(stop_sample_offset, len(samples))

            for sample in samples[start_ii:stop_ii]:
                # 转换通用数据格式
                for k in sample.keys():
                    if k in ['image_url', 'image_file']:
                        continue

                    # 对于list数据直接转换为numpy
                    if isinstance(sample[k], list):
                        if len(sample[k]) > 0:
                            # 如果存在二级list，则需要兼容空情况
                            if isinstance(sample[k][0], list):
                                max_num = 0
                                for i in range(len(sample[k])):
                                    if max_num < len(sample[k][i]):
                                        max_num = len(sample[k][i])
                                
                                for i in range(len(sample[k])):
                                    if len(sample[k][i]) == 0:
                                        sample[k][i] = [0 for _ in range(max_num)]

                        # 转换到numpy类型    
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
                    image_path = os.path.join(src_folder, sample['image_file'])
                    if os.path.exists(image_path):
                        with open(image_path, 'rb') as fp:
                            image_content = fp.read()
                            sample['image'] = image_content
                    else:
                        logging.error(f'Missing image {image_path}')
                        sample['image'] = b''

                if sample['image'] == b'':
                    # 图像为空时，直接忽略样本
                    logging.error(f"Sample {sample['image_file']} data abnormal")
                    continue
                else:
                    # yield sample
                    data_cache.put(sample)

        # 设置线程标记
        self.thread_stop_flag[thread_id] = 0

    def __len__(self):
        return self.num_sample

    def __iter__(self):
        while np.sum(self.thread_stop_flag) > 0:
            try:
                data = self.data_cache.get(2)   # 最大等待5s
            except:
                continue

            if data is not None:
                yield data


def package_to_kv(src_file, tgt_folder, prefix, size_in_shard=-1, **kwargs):
    # src_file json 文件 (仅支持标准格式 sample_gt.json)
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # 创建writer 实例
    kvw = KVDataWriter(prefix, tgt_folder, -1)

    # 写数据
    kvw.write(__SampleDataGenerator(src_file, thread_num=kwargs.get('thread_num', 10)))


def package_to_tfrecord(src_file, tgt_folder, prefix, size_in_shard=-1, **kwargs):
    # src_file json 文件 (仅支持标准格式 sample_gt.json)
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # 创建tfdatawriter 实例
    if size_in_shard <= 0:
        size_in_shard = 100000
    tfw = TFDataWriter(prefix, tgt_folder, size_in_shard)

    # 写数据
    tfw.write(__SampleDataGenerator(src_file, thread_num=kwargs.get('thread_num', 10)))


# test tfrecord
# with open('/root/workspace/extract/ss.json', 'r') as fp:
#     content = json.load(fp)
#     # content = set(content)
#     print(len(content))

# start_time = time.time()
# package_to_tfrecord("/root/workspace/dataset/finetune_4/annotation.json", "/root/workspace/dataset/extract", "hello", 100000, thread_num=3)
# print(f'all time {time.time() - start_time}')

# print('sdf')

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