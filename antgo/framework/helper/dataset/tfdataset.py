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
import threading
import json


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
                 data_path_list: typing.List[str],
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
                 sample_num_equalizer=True
                 ) -> None:
        super().__init__()
        self.data_path_list = data_path_list
        self.index_path_list = []
        for tfrecord_file in self.data_path_list:
            folder = os.path.dirname(tfrecord_file)
            index_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['index'])
            index_file = os.path.join(folder, index_file)
            self.index_path_list.append(index_file)

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
            if self.ratios is not None:
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

        num_samples_list = []
        self.num_samples = 0
        for i, index_path in enumerate(self.index_path_list):
            index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
            self.num_samples += len(index)
            num_samples_list.append((i, len(index)))        
        num_samples_list.sort(reverse=True, key=lambda x: x[1])
        self.real_num_samples = self.num_samples

        rank, world_size = get_dist_info()
        if world_size > 1:
            # TODO，现在多卡实现基于文件级别的拆分，粒度较粗
            assert(len(self.data_path_list) >= world_size)
            
            use_data_path_index_list = [num_samples_list[i][0] for i in range(rank, len(num_samples_list), world_size)]
            use_data_path_num_list = [num_samples_list[i][1] for i in range(rank, len(num_samples_list), world_size)]  
            self.real_num_samples = np.sum(use_data_path_num_list)

            self.num_samples = 0
            for rank_i in range(world_size):
                num = np.sum([num_samples_list[i][1] for i in range(rank_i, len(num_samples_list), world_size)])
                if self.num_samples < num:
                    self.num_samples = num

            self.data_path_list = [self.data_path_list[i] for i in use_data_path_index_list]
            self.index_path_list = [self.index_path_list[i] for i in use_data_path_index_list]
            if self.ratios is not None:
                self.ratios = [self.ratios[i] for i in use_data_path_index_list]

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
        new_sample = {}
        for k in sample.keys():
            if not k.startswith('__'):
                if self.raw_description[k] == 'numpy':
                    dtype = numpy_dtype_map[sample[f'__{k}_type'][0]]
                    shape = tuple(sample[f'__{k}_shape'])
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
            for transform in self.pipeline:
                sample = transform(sample)
                
            # arange warp
            sample = self._arrange(sample, self._fields, self._alias)        
            return sample

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        remain_sample_num = 0
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            remain_sample_num = int(self.num_samples - self.real_num_samples)
            if worker_info.num_workers != 1:
                if worker_info.id != worker_info.num_workers - 1:
                    remain_sample_num = remain_sample_num // worker_info.num_workers
                else:
                    remain_sample_num = (remain_sample_num // worker_info.num_workers) + (remain_sample_num - remain_sample_num // worker_info.num_workers * worker_info.num_workers)
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
                for data_path, index_path in zip(self.data_path_list, self.index_path_list)]

        it = None
        if self.ratios is not None and len(self.ratios) > 0:
            it = _sample_iterators(loaders, self.ratios, self.infinite, remain_sample_num)
        else:
            it = _order_iterators(loaders)

        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)

        it = map(self.__transform, it)
        return it

    def __len__(self):
        return self.num_samples
