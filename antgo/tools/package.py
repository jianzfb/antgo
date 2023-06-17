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
import copy


class __SampleDataGenerator(object):
    def __init__(self, json_files, thread_num=1, mode='json') -> None:
        self.json_file_list = json_files.split(',')
        self.mode = mode    # line or json. line: 意味着每一行是一个json格式，json: 意味着整体是一个json格式

        # 预先加载，获得样本总数
        self.num_sample = 0
        self.num_sample_offset = [0]
        self.num_sample_list = []
        for json_file in self.json_file_list:
            if self.mode == 'json':
                with open(json_file, 'r') as fp:
                    content = json.load(fp)
                    self.num_sample += len(content)
                    self.num_sample_offset.append(self.num_sample_offset[-1]+len(content))
                    self.num_sample_list.append(len(content))
            else:
                with open(json_file, 'r') as fp:
                    content = fp.readline()
                    content = content.strip()

                    num = 0
                    while content:
                        num += 1
                        content = fp.readline()
                        content = content.strip()
                        if content == '':
                            break

                    self.num_sample += num
                    self.num_sample_offset.append(self.num_sample_offset[-1]+num)
                    self.num_sample_list.append(num)

        print(f"Sample num {self.num_sample}")
        
        # 每个线程 分配的样本数
        assign_sample_num = (self.num_sample + (thread_num - 1)) // thread_num

        # 线程标记
        self.thread_stop_flag = [1 for _ in range(thread_num)]
        # 启动线程池进行数据生产
        # 数据队列 (队列最大容量10000)
        self.data_cache = Queue(100)        
        self.thread_pool = [
            threading.Thread(target=self.produce, args=(i*assign_sample_num, min((i+1)*assign_sample_num, self.num_sample), i, self.data_cache)) for i in range(thread_num)]

        # 启动线程
        for thread_i in range(thread_num):
            self.thread_pool[thread_i].start()

    def __process_sample(self, src_folder, sample, data_cache):
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
                else:
                    # 空list处理方法
                    sample[k] = np.array([], dtype=np.float32)

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
        if 'image_url' not in sample and 'image_file' not in sample:
            logging.error('dont have image_url and image_file')
            return
        
        if sample['image_url'] == '' and sample['image_file'] == '':
            logging.error('dont have image_url and image_file')
            return
        
        if sample['image_url'] != '':
            wait_count = 5
            while wait_count > 0:
                try:
                    pic = requests.get(sample['image_url'], timeout=20)
                    if pic.status_code != 200:
                        sample['image'] = b''
                        wait_count -= 1
                        time.sleep(2)
                        continue

                    sample['image'] = pic.content
                    break
                except:
                    logging.error("Couldnt download %s."%sample['image_url'])
                    sample['image'] = b''
                    wait_count -= 1
                    time.sleep(2)               
        else:
            try:
                image_path = os.path.join(src_folder, sample['image_file'])
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as fp:
                        image_content = fp.read()
                        sample['image'] = image_content
                else:
                    logging.error(f'Missing image {image_path}')
                    sample['image'] = b''
            except:
                logging.error("Failt load image_path")
                sample['image'] = b''

        if sample['image'] == b'':
            # 图像为空时，直接忽略样本
            logging.error(f"Sample {sample['image_file']} data abnormal")
            return
        else:
            # yield sample
            data_cache.put(sample)

    def line_mode_produce(self, start_ii, stop_ii, json_file, data_cache, src_folder):
        with open(json_file, 'r') as fp:
            content = fp.readline()
            content = content.strip()

            sample_i = 0
            while content:
                if sample_i < start_ii:
                    # 跳过样本
                    content = fp.readline()
                    content = content.strip()
                    sample_i += 1
                    if content == '':
                        break        
                    continue
                if sample_i >= stop_ii:
                    # 退出
                    break

                # 格式化一条样本
                sample = json.loads(content)
                
                # 读取下一条样本数据
                content = fp.readline()
                content = content.strip()
                sample_i += 1
                if content == '':
                    break

                self.__process_sample(src_folder, sample, data_cache)
    
    def json_mode_produce(self, start_ii, stop_ii, json_file, data_cache, src_folder):
        with open(json_file, 'r', encoding="utf-8") as fp:
            samples = json.load(fp)[start_ii:stop_ii]

        num = len(samples)
        for sample_i in range(num):
            sample = copy.deepcopy(samples[sample_i])

            self.__process_sample(src_folder, sample, data_cache)

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

            start_ii = 0
            stop_ii = self.num_sample_list[file_i]
            if file_i == start_file_i:
                start_ii = start_sample_offset

            if file_i == stop_file_i:
                stop_ii = min(stop_sample_offset, self.num_sample_list[file_i])

            print(f'thread_id {thread_id} start_ii {start_ii} stop_ii {stop_ii} in file_i {file_i}')
            if self.mode == 'line':
                self.line_mode_produce(start_ii, stop_ii, json_file, data_cache, src_folder)
            else:
                self.json_mode_produce(start_ii, stop_ii, json_file, data_cache, src_folder)

        # 设置线程标记
        self.thread_stop_flag[thread_id] = 0

    def __len__(self):
        return self.num_sample

    def __iter__(self):
        while np.sum(self.thread_stop_flag) > 0:
            try:
                data = self.data_cache.get(True, timeout=5)   # 最大等待5s
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
    kvw.write(__SampleDataGenerator(src_file, thread_num=kwargs.get('thread_num', 10), mode=kwargs.get('mode', 'json')))


def package_to_tfrecord(src_file, tgt_folder, prefix, size_in_shard=-1, **kwargs):
    # src_file json 文件 (仅支持标准格式 sample_gt.json)
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # 创建tfdatawriter 实例
    if size_in_shard <= 0:
        size_in_shard = 100000
    tfw = TFDataWriter(prefix, tgt_folder, size_in_shard)

    # 写数据
    tfw.write(__SampleDataGenerator(src_file, thread_num=kwargs.get('thread_num', 10), mode=kwargs.get('mode', 'json')))


# test tfrecord
# with open('/root/workspace/extract/ss.json', 'r') as fp:
#     content = json.load(fp)
#     # content = set(content)
#     print(len(content))

# start_time = time.time()
# package_to_tfrecord("/root/workspace/shuffle-hard-mining-v5/annotation_label_0.json,/root/workspace/shuffle-hard-mining-v5/annotation_label_0.json,/root/workspace/shuffle-hard-mining-v5/annotation_label_0.json", "/root/workspace/dataset/extract", "hello", 100000, thread_num=2, mode='json')
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