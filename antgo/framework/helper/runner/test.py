# Copyright (c) MLTALKER. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import os

import torch
import torch.distributed as dist
from .dist_utils import get_dist_info
from antgo.framework.helper.utils import *


def single_gpu_test(model, data_loader, needed_info=None):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = {}
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            assert(type(data) == dict)
            data.update({
                    'return_loss': False
                })
            result = model(**data)

        # result is dict
        batch_size = 0
        for var_name, var_value in result.items():
            if var_name not in results:
                results[var_name] = []
            
            if batch_size == 0:
                batch_size = len(var_value)
            results[var_name].extend([v.cpu().numpy() if type(v) == torch.Tensor else v for v in var_value])

        if needed_info is not None:
            for key in needed_info:
                if key not in results:
                    results[key] = []
                
                results[key].extend([v.cpu().numpy() if type(v) == torch.Tensor else v for v in data[key]])
        
        for _ in range(batch_size):
            prog_bar.update()
    
    rearrange_results = []
    for var_name, var_value in results.items():
        if len(rearrange_results) == 0:
            rearrange_results = [{} for _ in range(len(var_value))]
        
        for i in range(len(var_value)):
            rearrange_results[i][var_name] = var_value[i]

    return rearrange_results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, needed_info=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = {}
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader): 
        with torch.no_grad():
            if type(data) == list or type(data) == tuple:         
                result = model(*data, return_loss=False)
            else:
                data.update({
                    'return_loss': False
                })                       
                result = model(**data)

        # result is dict
        batch_size = 0
        for var_name, var_value in result.items():
            if var_name not in results:
                results[var_name] = []
            
            if batch_size == 0:
                batch_size = len(var_value)
            results[var_name].extend([v.cpu().numpy() if type(v) == torch.Tensor else v for v in var_value])

        if needed_info is not None:
            for key in needed_info:
                if key not in results:
                    results[key] = []
                
                results[key].extend([v.cpu().numpy() if type(v) == torch.Tensor else v for v in data[key]])

        if rank == 0:
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        # 暂不启用
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            if not os.path.exists('.dist_test'):
                os.makedirs('.dist_test', exist_ok=True)

            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)

    # dump the part result to the dir
    print('save part before')
    try:
        with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as fp:
            pickle.dump(result_part, fp)
    except Exception as e:
        print(e)
    print('save part after')

    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_dict = {}
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            with open(part_file, 'rb') as fp:
                # part_result = mmcv.load(part_file)
                part_result = pickle.load(fp)

            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            for k,v in part_result.items():
                if k not in part_dict:
                    part_dict[k] = []
                part_dict[k].append(v)

        # sort the results
        ordered_results = {}
        for k,v in part_dict.items():
            if k not in ordered_results:
                ordered_results[k] = []
            for res in zip(*v):
                ordered_results[k].extend(list(res))
            ordered_results[k] = ordered_results[k][:size]

        # remove tmp dir
        shutil.rmtree(tmpdir)

        rearrange_results = []
        for var_name, var_value in ordered_results.items():
            if len(rearrange_results) == 0:
                rearrange_results = [{} for _ in range(len(var_value))]
            
            for i in range(len(var_value)):
                rearrange_results[i][var_name] = var_value[i]

        return rearrange_results


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
