from email.utils import localtime
import logging
import sys
import numpy as np
import torch
import torch.distributed as dist
from antgo.framework.helper.utils import build_from_cfg
from antgo.framework.helper.runner.dist_utils import get_dist_info
from tfrecord.reader import *
from tfrecord import iterator_utils
from antgo.framework.helper.dataset.builder import DATASETS
import copy
from antgo.dataflow.datasetio import *
from antgo.framework.helper.fileio.file_client import *
import threading
import json
from pprint import pprint
from filelock import FileLock

def _cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element

def _sample_iterators(iterators, ratios, infinite, remain_sample_num):
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.
    
    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    ext_iterators = None
    iterator_num = len(iterators)
    if infinite:
        iterators = [_cycle(iterator) for iterator in iterators]
    else:
        ext_iterators = [_cycle(iterator) for iterator in iterators]
        iterators = [iterator() for iterator in iterators]
    
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
        
    in_remain_sample_mode = False
    while iterators or in_remain_sample_mode:
        try:
            if not in_remain_sample_mode:
                choice = np.random.choice(len(ratios), p=ratios)
                yield next(iterators[choice])
            else:
                if remain_sample_num == 0:
                    in_remain_sample_mode = False
                    break
                remain_sample_num -= 1

                yield next(ext_iterators[np.random.randint(0, iterator_num)])
        except StopIteration:
            if iterators:
                del iterators[choice]
                ratios = np.delete(ratios, choice)
                ratios = ratios / ratios.sum()

                if len(iterators) == 0 and remain_sample_num > 0 and not in_remain_sample_mode:
                    in_remain_sample_mode = True


def _order_iterators(iterators):
    iterators = [iterator() for iterator in iterators]
    choice = 0
    while iterators:
        try:
            yield next(iterators[choice])
        except StopIteration:
            if iterators:
                del iterators[choice]
                choice += 1


@DATASETS.register_module()
class TFDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path_list: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """
    def __init__(self,
                 data_folder,
                 ratios: typing.Union[typing.List[float], None]=None,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = 1024,
                 pipeline: typing.Optional[typing.List]=None, 
                 weak_pipeline: typing.Optional[typing.List]=None, 
                 strong_pipeline: typing.Optional[typing.List]=None, 
                 sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 compression_type: typing.Optional[str] = None,
                 infinite: typing.Optional[bool] = False,
                 inputs_def=None,
                 sample_num_equalizer=True,
                 auto_ext_info=[]) -> None:
        super().__init__()
        if isinstance(data_folder, str):
            data_folder = [data_folder]

        # 准备数据
        self._prepare_data(data_folder)

        if description is None:
            description = {}
            print('Using default tfrecord description.')
            tfdataset_file_path = os.path.realpath(__file__)
            parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(tfdataset_file_path))))
            with open(os.path.join(parent_folder, 'resource', 'templates', 'sample_gt.json'), 'r') as fp:
                key_map_info = json.load(fp)

            for k in key_map_info.keys():                
                # 对于list数据直接转换为numpy
                if isinstance(key_map_info[k], list):
                    description[k] = 'numpy'
                # 对于dict数据转换成字符串
                elif isinstance(key_map_info[k], dict):
                    description[k] = 'dict'
                # 对bool数据转换成int
                elif isinstance(key_map_info[k], bool):
                    description[k] = 'int'
                elif isinstance(key_map_info[k], int):
                    description[k] = 'int'
                elif isinstance(key_map_info[k], float):
                    description[k] = 'float'
                elif isinstance(key_map_info[k], str):
                    description[k] = 'str'
                elif isinstance(key_map_info[k], bytes):
                    description[k] = 'byte'
                else:
                    print('unkown data type')

            # 默认字段，用于存储图像
            description['image'] = 'byte'

        self.description = {}
        self.raw_description = description
        for k, v in description.items():
            if v == 'numpy':
                self.description.update({
                    k: 'byte',
                    f'__{k}_type': 'int',
                    f'__{k}_shape': 'int'
                })
            elif v == 'str':
                self.description.update({
                    k: 'byte'
                })
            elif v == 'dict':
                self.description.update({
                    k: 'byte'
                })
            else:
                self.description.update({
                    k: v
                })

        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.sample_num_equalizer = sample_num_equalizer
        self.compression_type = compression_type
        self.ratios = ratios
        if shuffle_queue_size > 0:
            # 如果已经设置了shuffle queue，则设置默认ratios
            if self.ratios is None:
                self.ratios = [1 for _ in range(len(self.data_path_list))]
        self._fields = copy.deepcopy(inputs_def['fields']) if inputs_def else None
        self._alias = None
        if self._fields is not None and 'alias' in inputs_def:
            self._alias = copy.deepcopy(inputs_def['alias'])

        if self._fields is not None:
            if self._alias is None:
                self._alias = copy.deepcopy(self._fields)

        self.pipeline = []
        self.weak_pipeline = []
        self.strong_pipeline = []
        if pipeline is not None:
            from antgo.framework.helper.dataset import PIPELINES
            for transform in pipeline:
                if isinstance(transform, dict):
                    transform = build_from_cfg(transform, PIPELINES)
                    self.pipeline.append(transform)
                else:
                    raise TypeError('pipeline must be a dict')

            if weak_pipeline is not None and strong_pipeline is not None:
                for transform in weak_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.weak_pipeline.append(transform)
                    else:
                        raise TypeError('weak_pipeline must be a dict')
                
                for transform in strong_pipeline:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        self.strong_pipeline.append(transform)
                    else:
                        raise TypeError('strong_pipeline must be a dict')
        self.infinite = infinite

        # 自动扩展样本信息
        self.auto_ext_info = auto_ext_info
        self.epoch = 10   
        self.sample_id = 0      # 先默认为0

        self.num_samples_list = []
        self.num_samples = 0
        for i, index_path in enumerate(self.index_path_list):
            index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
            self.num_samples += len(index) 
            self.num_samples_list.append(len(index))

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.select_index_list_in_world = []    # format: [[],[],[],...]
        if world_size > 1:
            # TODO，现在多卡实现基于文件级别的拆分，粒度较粗
            assert(len(self.data_path_list) >= world_size)

            # 公平选择，尽量确保每张卡有相同的样本数量
            self.select_index_list_in_world = self._fair_select(self.num_samples_list, world_size)

            # 每张卡期望使用的样本数（以最大为准，不足的通过后续重复采样补充）
            self.num_samples = 0
            for rank_i in range(world_size):
                num = np.sum([self.num_samples_list[i] for i in self.select_index_list_in_world[rank_i]])
                if self.num_samples < num:
                    self.num_samples = num

    def _select_next_set(self, num_samples_list, target_num):
        data = num_samples_list
        dp = [[0 for i in range(target_num+1)] for _ in range(len(data)+1)]

        for i in range(1,len(data)+1):
            for j in range(0, target_num+1):
                if j >= data[i-1]:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-data[i-1]]+data[i-1])
                    pass
                else:
                    dp[i][j] = dp[i-1][j]
        
        select_list = []
        for i in range(len(data), 0, -1):
            if dp[i][j] > dp[i-1][j]:
                select_list.append(i-1)
                j -= data[i-1]

        return select_list
    
    def _fair_select(self, num_samples_list, world_size):
        select_index_list_in_world = []

        index_list = list(range(len(num_samples_list)))     # 维持全局索引
        num_samples_list = copy.deepcopy(num_samples_list)
        
        for i in range(world_size):
            if i == world_size - 1:
                select_index_list_in_world.append(index_list)
                break

            # 每次进行平均分配时，重新计算平均分配目标
            target_num = (sum(num_samples_list) + world_size-i-1)  // (world_size-i)
            select_local_index = self._select_next_set(num_samples_list, target_num)

            # 如果发现，本次分配方案，会导致剩余样本无法至少每个slot一个样本的话需要强制减少本次分配
            if len(index_list) - len(select_local_index) < world_size-i-1:
                remain_select_num = len(index_list) - (world_size-i-1)
                select_local_index = select_local_index[:remain_select_num]

            select_index = [index_list[j] for j in select_local_index]            
            select_index_list_in_world.append(select_index)

            num_samples_list = [num_samples_list[j] for j in range(len(num_samples_list)) if j not in select_local_index]               
            index_list = [k for k in index_list if k not in select_index]        

        return select_index_list_in_world

    def _arrange(self, sample, fields, alias):
        if fields is None:
            return sample      
          
        if type(fields[0]) == list or type(fields[0]) == tuple:
            warp_ins = []
            for alia, field in zip(alias, fields):
                one_ins = {}
                for aa, ff in zip(alia, field):
                    one_ins[aa] = sample[ff]
                
                warp_ins.append(one_ins)
            return warp_ins
        
        warp_ins = {}
        for alia, field in zip(alias, fields):
            warp_ins[alia] = sample[field]

        return warp_ins

    def __transform(self, sample):
        # 转换样本
        new_sample = {}
        for k in sample.keys():
            if not k.startswith('__'):
                if self.raw_description[k] == 'numpy':
                    dtype = numpy_dtype_map[sample[f'__{k}_type'][0]]
                    shape = tuple(sample[f'__{k}_shape'])
                    if isinstance(sample[k], bytes):
                        new_sample[k] = np.frombuffer(bytearray(sample[k]), dtype=dtype).reshape(shape).copy()
                    else:
                        new_sample[k] = np.frombuffer(bytearray(sample[k].tobytes()), dtype=dtype).reshape(shape).copy()
                elif self.raw_description[k] == 'str':
                    new_sample[k] = sample[k].tobytes().decode('utf-8')
                elif self.raw_description[k] == 'dict':
                    new_sample[k] = json.loads(sample[k].tobytes().decode('utf-8'))
                else:
                    new_sample[k] = sample[k]
        sample = new_sample
        weak_sample = None
        strong_sample = None
        if len(self.weak_pipeline) > 0 or len(self.strong_pipeline) > 0:
            # 扩展样本信息
            for ext_key in self.auto_ext_info:
                if ext_key not in sample:
                    sample[ext_key] = f'{self.sample_id}'

            weak_sample = copy.deepcopy(sample)
            for transform in self.weak_pipeline:
                weak_sample = transform(weak_sample)

            strong_sample = copy.deepcopy(weak_sample)
            for transform in self.strong_pipeline:
                strong_sample = transform(strong_sample)

        if weak_sample is not None and strong_sample is not None:
            for transform in self.pipeline:
                weak_sample = transform(weak_sample)

            for transform in self.pipeline:
                strong_sample = transform(strong_sample)

            weak_sample = self._arrange(weak_sample, self._fields, self._alias)
            strong_sample = self._arrange(strong_sample, self._fields, self._alias)
            return [weak_sample, strong_sample]
        else:
            # 扩展样本信息
            for ext_key in self.auto_ext_info:
                if ext_key not in sample:
                    sample[ext_key] = f'{self.sample_id}'

            for transform in self.pipeline:
                sample = transform(sample)

            # arange warp       
            sample = self._arrange(sample, self._fields, self._alias)        
            return sample

    def __iter__(self):
        # 获得线程信息
        worker_info = torch.utils.data.get_worker_info()

        # 每个epoch后，会重新调用__iter__
        # WARN: 对于多卡训练环境，在每次epoch时，重新编排数据文件的分配
        # 先按照单卡配置，进行赋初值
        real_num_samples = self.num_samples     # 单卡下期望的样本总数就是真实样本总数（只有多卡下才涉及补齐）
        ratios = self.ratios                    # 每个数据集的采样率
        data_path_list = self.data_path_list    # 总数据文件tfrecord列表
        index_path_list = self.index_path_list  # 总数据文件index列表
        if len(self.select_index_list_in_world) > 0:
            # 使用re_init_count进行shuffle （每个进程每个线程面对的是相同的re_init_count）
            local_select_index_list_in_world = copy.deepcopy(self.select_index_list_in_world)
            np.random.seed(self.epoch)  # 注意seed仅对当前线程有效，所以可以放心使用
            np.random.shuffle(local_select_index_list_in_world)

            # 选出当前卡，在本次epoch下使用的数据集
            # 每个线程，应该拥有相同的shuffle
            select_index_list = copy.deepcopy(local_select_index_list_in_world[self.rank])
            np.random.shuffle(select_index_list)
            
            # 基于选择的索引，获得具体样本数量列表
            use_data_path_num_list = [self.num_samples_list[i] for i in select_index_list]
            real_num_samples = np.sum(use_data_path_num_list)

            # 基于选择的索引，获得文件列表
            data_path_list = [self.data_path_list[i] for i in select_index_list]
            index_path_list = [self.index_path_list[i] for i in select_index_list]

            # 基于选择的索引，获得采样率列表
            if self.ratios is not None:
                ratios = [self.ratios[i] for i in select_index_list]

            # debug
            pprint(f'Rank {self.rank} (TOTAL NUM {real_num_samples} TARGET NUM {self.num_samples})')
            pprint(f'Rank {self.rank} local-index {local_select_index_list_in_world}')
            pprint(f'Rank {self.rank} thread id {worker_info.id} use index {select_index_list} and data path list {data_path_list}')

        remain_sample_num = 0
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            if worker_info.num_workers != 1:
                expect_target_num = \
                    self.num_samples * (worker_info.id+1) // worker_info.num_workers - \
                    self.num_samples * (worker_info.id) // worker_info.num_workers

                real_target_num = real_num_samples * (worker_info.id+1) // worker_info.num_workers - \
                    real_num_samples * (worker_info.id) // worker_info.num_workers

                remain_sample_num = expect_target_num - real_target_num

            # debug
            pprint(f'Rank {self.rank} thread {worker_info.id} face remain sample_num {remain_sample_num}')
        else:
            shard = None

        if not self.sample_num_equalizer:
            remain_sample_num = 0

        loaders = [functools.partial(tfrecord_loader, data_path=data_path,
                                    index_path=index_path,
                                    shard=shard,
                                    description=self.description,
                                    sequence_description=self.sequence_description,
                                    compression_type=self.compression_type,
                                    )
                for data_path, index_path in zip(data_path_list, index_path_list)]

        it = None
        if ratios is not None and len(ratios) > 0:
            it = _sample_iterators(loaders, ratios, self.infinite, remain_sample_num)
        else:
            it = _order_iterators(loaders)

        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)

        it = map(self.__transform, it)
        return it

    def __len__(self):
        return self.num_samples

    def _prepare_data(self, dataset_folders):
        self.data_path_list = []
        self.index_path_list = []
        for dataset_folder in dataset_folders:
            if ':///' in dataset_folder:
                # 远程存储模式
                terms = dataset_folder.split('/')
                dataset_name = terms[-1]
                if terms[-1] == '*' or terms[-1] == '':
                    dataset_name = terms[-2]

                # 远程存储，自动下载
                local_temp_folder = f'./temp_{dataset_name}'

                # 多进程锁
                lock = FileLock('DATASET.lock')
                with lock:
                    if not os.path.exists(local_temp_folder):
                        # 数据集不存在，需要重新下载，并创建标记
                        # 创建临时目录
                        os.makedirs(local_temp_folder)

                        # 下载
                        file_client_get(dataset_folder, local_temp_folder)
            else:
                # 本地存储模式
                if '*' in dataset_folder:
                    dataset_folder = dataset_folder.replace('*', '')

            # 替换为本地路径
            if ':///' in dataset_folder:
                terms = dataset_folder.split('/')
                dataset_name = terms[-1]
                if terms[-1] == '*' or terms[-1] == '':
                    dataset_name = terms[-2]

                # 使用本地地址替换
                dataset_folder = f'./temp_{dataset_name}'

            # 遍历文件夹，发现所有tfrecord数据
            part_path_list = []
            for tfrecord_file in os.listdir(dataset_folder):
                if tfrecord_file.endswith('tfrecord'):
                    tfrecord_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['tfrecord'])
                    part_path_list.append(f'{dataset_folder}/{tfrecord_file}')

            part_index_path_list = []
            for i in range(len(part_path_list)):
                tfrecord_file = part_path_list[i]
                folder = os.path.dirname(tfrecord_file)
                if tfrecord_file.endswith('tfrecord') or tfrecord_file.endswith('index'):
                    index_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['index'])
                    index_file = f'{folder}/{index_file}'
                    tfrecord_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['tfrecord'])
                    part_path_list[i] = f'{folder}/{tfrecord_file}'
                else:
                    index_file = tfrecord_file+'-index'
                    part_path_list[i] = tfrecord_file+'-tfrecord'
                part_index_path_list.append(index_file)
            
            self.data_path_list.extend(part_path_list)
            self.index_path_list.extend(part_index_path_list)


# data = [2,4,6,7,8,5]
# data_sum = int(np.sum(data))

# dp = [[0 for i in range(data_sum//2+1)] for _ in range(len(data)+1)]

# for i in range(1,len(data)+1):
#     for j in range(0, data_sum//2+1):
#         if j >= data[i-1]:
#             dp[i][j] = max(dp[i-1][j], dp[i-1][j-data[i-1]]+data[i-1])
#             pass
#         else:
#             dp[i][j] = dp[i-1][j]

# j = data_sum//2
# print(j)

# print('select')
# for i in range(len(data), 0, -1):
#     if dp[i][j] > dp[i-1][j]:
#         print(i-1)
#         j -= data[i-1]

# print('sdf')

# abcd = TFDataset(data_path_list=[
#     '/root/workspace/dataset/hand-cls/yongchun_hand_gesture-00000-of-00003-tfrecord',
#     '/root/workspace/dataset/hand-cls/yongchun_hand_gesture-00001-of-00003-tfrecord',
#     '/root/workspace/dataset/hand-cls/yongchun_hand_gesture-00002-of-00003-tfrecord',
#     '/root/workspace/dataset/hand-cls/yongchun-cls-23-00000-of-00001-tfrecord']
# )
# print('sd')
# select_all = abcd._fair_select([2,4,6,7,8,5], 3)
# print(select_all)

# for ii in select_all:
#     print(np.sum(np.array([2,4,6,7,8,5])[ii]))