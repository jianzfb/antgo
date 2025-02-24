import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
from numpy.random.mtrand import sample
import torch
from torch.utils.data.dataset import IterableDataset
from antgo.framework.helper.parallel import collate
from antgo.framework.helper.runner import get_dist_info
from antgo.framework.helper.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader


from .samplers import (ClassAwareSampler, DistributedGroupSampler,
                       DistributedSampler, GroupSampler, InfiniteBatchSampler,
                       InfiniteGroupBatchSampler, KVSampler, DistributedKVSampler)

# if platform.system() != 'Windows':
#     # https://github.com/pytorch/pytorch/issues/973
#     import resource
#     rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#     base_soft_limit = rlimit[0]
#     hard_limit = rlimit[1]
#     soft_limit = min(max(4096, base_soft_limit), hard_limit)
#     resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset, IterConcatDataset, CircleDataset)
    if isinstance(cfg, (list, tuple)):
        type_list = [cfg[i].type for i in range(len(cfg)) if isinstance(cfg[i], dict)]
        if 'TFDataset' in type_list:
            dataset = IterConcatDataset([build_dataset(c, default_args) for c in cfg])
        else:
            dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            pipeline=cfg.get('pipeline', None),
            inputs_def=cfg.get('inputs_def', None))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'CircleDataset':
        dataset = CircleDataset(
            build_dataset(cfg['dataset'], default_args), cfg['sample_num']
        )
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     class_aware_sampler=None,
                     prefetch_factor=2,                # 预加载prefetch_factor * num_workers samples prefetched across all workers
                     ignore_stack=[],
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        class_aware_sampler (dict): Whether to use `ClassAwareSampler`
            during training. Default: None.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu)
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu)
        num_workers = num_gpus * workers_per_gpu

    strategy = None
    if isinstance(samples_per_gpu, list):
        strategy = {}
        for index, s in enumerate(samples_per_gpu):
            strategy[index] = s

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset,
                batch_size,
                world_size,
                rank,
                seed=seed,
                shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if class_aware_sampler is not None:
            # ClassAwareSampler can be used in both distributed and
            # non-distributed training.
            num_sample_class = class_aware_sampler.get('num_sample_class', 1)
            sampler = ClassAwareSampler(
                dataset,
                samples_per_gpu,
                world_size,
                rank,
                seed=seed,
                num_sample_class=num_sample_class)
        elif dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, 
                    samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu),
                    world_size, 
                    rank, 
                    seed=seed, 
                    strategy=strategy)

            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed, strategy=strategy)
        else:
            sampler = GroupSampler(
                dataset,
                samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu),
                strategy=strategy) if shuffle else None

        batch_sampler = None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        prefetch_factor=prefetch_factor,            # 预读取配置
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu, ignore_stack=ignore_stack),
        pin_memory=kwargs.pop('pin_memory', True),  # 固定内存并启用从主机到 GPU 的更快和异步内存复制
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def build_kv_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=0,
                     runner_type='EpochBasedRunner',
                     prefetch_factor=10,                # 预加载prefetch_factor * num_workers samples prefetched across all workers
                     persistent_workers=False,
                     ignore_stack=[],
                     **kwargs):
    sampler = None
    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu)
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * (samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu))
        num_workers = num_gpus * workers_per_gpu

    strategy = None
    if isinstance(samples_per_gpu, list):
        strategy = {}
        for index, s in enumerate(samples_per_gpu):
            strategy[index] = s
    
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedKVSampler(
            dataset,
            batch_size, 
            num_replicas=world_size,
            shuffle=shuffle,
            rank=rank,
            seed=seed, 
            drop_last=kwargs.get('drop_last', False), 
            strategy=strategy)
    else:
        sampler = \
            KVSampler(
                dataset, 
                samples_per_gpu if not isinstance(samples_per_gpu, list) else np.sum(samples_per_gpu), 
                drop_last=kwargs.get('drop_last', False), 
                shuffle=shuffle,
                strategy=strategy)

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
        
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        batch_sampler=None,
        pin_memory=kwargs.pop('pin_memory', True),
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu, ignore_stack=ignore_stack),
        worker_init_fn=init_fn)

    return data_loader


def build_iter_dataloader(dataset,   
                          samples_per_gpu,
                          workers_per_gpu,
                          ignore_stack=[],
                          prefetch_factor=10,       # 预加载prefetch_factor * num_workers samples prefetched across all workers
                          **kwargs):
    batch_size = samples_per_gpu if not isinstance(samples_per_gpu, list) else int(np.sum(samples_per_gpu))
    dataset.samples_per_gpu = samples_per_gpu
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=workers_per_gpu,
        prefetch_factor=prefetch_factor,
        pin_memory=kwargs.pop('pin_memory', True),
        collate_fn=partial(collate, samples_per_gpu=batch_size, ignore_stack=ignore_stack),
        drop_last=kwargs.get('drop_last', True))

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # 初始化
    worker_info = torch.utils.data.get_worker_info()
    if getattr(worker_info.dataset, 'worker_init_fn', None):
        worker_info.dataset.worker_init_fn(worker_id)