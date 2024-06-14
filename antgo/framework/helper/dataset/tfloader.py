import logging
import sys
import numpy as np
import torch
import torch.distributed as dist
from antgo.framework.helper.utils import build_from_cfg
from antgo.framework.helper.runner.dist_utils import get_dist_info
from tfrecord.reader import *
from tfrecord import iterator_utils
import copy
from antgo.dataflow.datasetio import *
from antgo.framework.helper.fileio.file_client import *
import json
from pprint import pprint
from filelock import FileLock
from antgo.framework.helper.dataset import *


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


class TFLoader(torch.utils.data.IterableDataset):
    def __init__(self,
                 data_folder,
                 ratios: typing.Union[typing.List[float], None]=None,
                 shuffle_queue_size: typing.Optional[int] = 1024,
                 infinite: typing.Optional[bool] = False,
                 sample_num_equalizer=True,
                 rank_size=1) -> None:
        super().__init__()
        if isinstance(data_folder, str):
            data_folder = [data_folder]

        # 准备数据
        self._prepare_data(data_folder)

        self.shuffle_queue_size = shuffle_queue_size
        self.sample_num_equalizer = sample_num_equalizer
        self.ratios = ratios
        if shuffle_queue_size > 0:
            # 如果已经设置了shuffle queue，则设置默认ratios
            if self.ratios is None:
                self.ratios = [1 for _ in range(len(self.data_path_list))]
        self.infinite = infinite

        # 自动扩展样本信息
        # 需要全局同步，获得样本列表并规划分配
        data_state = {}
        self.rank_size = rank_size
        self.num_samples_list = []
        self.num_samples = 0
        self.world_size = self.rank_size
        self.epoch = 0         # 外部设定

        # 分析数据信息
        self._analyze_data()

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

    def __iter__(self):
        # 获得线程信息
        worker_info = get_data_worker_info()

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
            select_index_list = copy.deepcopy(local_select_index_list_in_world[worker_info.rank])
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
        else:
            select_index_list = list(range(len(self.num_samples_list)))

        remain_sample_num = 0
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            if worker_info.num_workers != 1:
                expect_target_num = ((int)(self.num_samples) + (int)(2*worker_info.num_workers - 1))//worker_info.num_workers
                real_target_num = 0
                for select_i in select_index_list:
                    select_num_sample = self.num_samples_list[select_i]
                    real_target_num += \
                        (select_num_sample * (worker_info.id+1)) // worker_info.num_workers - \
                            (select_num_sample * (worker_info.id)) // worker_info.num_workers
                remain_sample_num = expect_target_num - real_target_num
            else:
                expect_target_num = 0
                remain_sample_num = self.num_samples - real_num_samples
        else:
            shard = None

        if not self.sample_num_equalizer:
            remain_sample_num = 0

        loaders = [functools.partial(tfrecord_iterator, data_path=data_path,
                                    index_path=index_path,
                                    shard=shard)
                for data_path, index_path in zip(data_path_list, index_path_list)]

        it = None
        if ratios is not None and len(ratios) > 0:
            it = _sample_iterators(loaders, ratios, self.infinite, remain_sample_num)
        else:
            it = _order_iterators(loaders)

        # if self.shuffle_queue_size:
        #     it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)

        return it

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
                
                if not os.path.exists(dataset_folder):
                    print(f'No {dataset_folder} in local disk.')
                    continue

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

    def __len__(self):
        return self.num_samples

    def _analyze_data(self):
        for i, index_path in enumerate(self.index_path_list):
            index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
            self.num_samples += len(index) 
            self.num_samples_list.append(len(index))

        self.select_index_list_in_world = []    # format: [[],[],[],...]
        if self.world_size > 1:
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


if __name__ == "__main__":
    dataset = TFLoader(
        '/workspace/dataset/personball/real_ext_crop_zhanhui',   
    )
    ds = DataServer('hwiaaabc', dataset, '192.168.1.90', 5672, consumer_size=2, worker_num=2, epoch_num=5)
    ds.start()
    print('lalala')