from numpy.random.mtrand import sample
from antgo.framework.helper.parallel import is_module_wrapper
from antgo.framework.helper.runner.hooks import HOOKS, Hook
from antgo.framework.helper.utils import Registry, build_from_cfg
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from bisect import bisect_right
from antgo.framework.helper.dataset import (build_dataloader, build_dataset)
from antgo.framework.helper.runner import BaseModule
from .samplingmethods.kcenter_greedy import *
import os
import json
import numpy as np
from scipy.stats import entropy
import time

UNCERTAINTY_SAMPLING = Registry('UNCERTAINTY_SAMPLING')

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)


@UNCERTAINTY_SAMPLING.register_module()
class UncertaintyByEntropy(object):
    def __init__(self, key, sampling_num=0) -> None:
        super().__init__()
        self.key = key
        self.sampling_num = sampling_num
    
    def __call__(self, results, dataset, sampling_num, pre_selected=None):
        entropy_list = []
        if pre_selected is None:
            pre_selected = list(range(len(results)))
        
        if self.sampling_num > 0:
            sampling_num = self.sampling_num
        for i in pre_selected:
            # hand_class_preds: NxBx4
            hand_class_preds = np.concatenate(results[i][self.key], 0)
            hand_class_preds = softmax(hand_class_preds, -1)
            hand_class_preds = np.mean(hand_class_preds, 0)
            # Bx4
            entropy_list.append((entropy(hand_class_preds), i))

        ordered_unlabeled = sorted(entropy_list, key=lambda x: x[0], reverse=True)
        ordered_unlabeled = ordered_unlabeled[:sampling_num]
        ordered_indexs = [s[1] for s in ordered_unlabeled]
        
        sample_list = []
        selected_list = []
        for index in ordered_indexs:
            # get index in dataset
            index = pre_selected[index]
            selected_list.append(index)

            # get data info at index
            info = dataset.get_ann_info(index)

            image_file = info['image_metas']['image_file']
            image_index = info['image_metas']['id']
            sample_list.append({
                'id': image_index,
                'image_file': image_file
            })

        return sample_list, selected_list
        

@UNCERTAINTY_SAMPLING.register_module()
class UncertaintyByKCenterEntropy(object):
    def __init__(self, key, sampling_num=0) -> None:
        super().__init__()
        self.key = key
        self.sampling_num = sampling_num
    
    def __call__(self, results, dataset, sampling_num, pre_selected=None):
        # 1.step stack feature
        feature_stack = []
        if pre_selected is None:
            pre_selected = list(range(len(results)))        
        for i in pre_selected:
            feature = results[i][self.key]
            if type(feature) == list:
                feature = np.mean(np.stack(feature,0),0)

            feature_stack.append(feature.flatten())
        feature_stack = np.stack(feature_stack, 0)

        # 2.step centergreedy
        if self.sampling_num > 0:
            sampling_num = self.sampling_num        
        kcenter_greedy = kCenterGreedy(feature_stack)
        selected_i_list = kcenter_greedy.select_batch(model=None, already_selected=[], N=sampling_num)

        # 3.step select sample
        sample_list = []
        selected_list = []
        for index in selected_i_list:
            # get index in dataset
            index = pre_selected[index]
            selected_list.append(index)
            # get data info at index
            info = dataset.get_ann_info(index)            

            image_file = info['image_metas']['image_file']
            image_index = info['image_metas']['id']
            sample_list.append({
                'id': image_index,
                'image_file': image_file
            })

        return sample_list, selected_list



@UNCERTAINTY_SAMPLING.register_module()
class UncertaintyByUniform(object):
    def __init__(self, sampling_num=0) -> None:
        super().__init__()
        self.sampling_num = sampling_num
    
    def __call__(self, results, dataset, sampling_num, pre_selected=None):
        # 1.step uniform sampling
        if pre_selected is None:
            pre_selected = list(range(len(results)))        

        if self.sampling_num > 0:
            sampling_num = self.sampling_num          
        selected_i_list = list(range(len(pre_selected)))
        np.random.shuffle(selected_i_list)
        selected_i_list = selected_i_list[:sampling_num]

        # 2.step select
        sample_list = []
        selected_list = []
        for index in selected_i_list:
            # get index in dataset
            index = pre_selected[index]
            selected_list.append(index)
            # get data info at index
            info = dataset.get_ann_info(index)            

            image_file = info['image_metas']['image_file']
            image_index = info['image_metas']['id']
            sample_list.append({
                'id': image_index,
                'image_file': image_file
            })

        return sample_list, selected_list


@UNCERTAINTY_SAMPLING.register_module()
class UncertaintyByComposer(object):
    def __init__(self, sampling_cfgs) -> None:
        super().__init__()
        self.sampling_fns= []
        for cfg in sampling_cfgs:
            sampling_fn = build_from_cfg(cfg, UNCERTAINTY_SAMPLING)
            self.sampling_fns.append(sampling_fn)

    def __call__(self, results, dataset, sampling_num, pre_selected=None):
        sample_list = []
        selected_list = None
        for sampling_fn in self.sampling_fns:
            sample_list, selected_list = sampling_fn(results, dataset, sampling_num, selected_list)

        return sample_list, selected_list


@HOOKS.register_module()
class ActiveLearningHook(Hook):
    def __init__(
        self, 
        data_cfg, 
        sampling_num, 
        out_dir='./activelearning/', 
        test_fn=None, 
        sampling_fn=None, 
        sampling_count=10, 
        is_dist=False) -> None:
        super().__init__()
        dataset = build_dataset(data_cfg['dataset'])
        dataloader_default_args = dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                dist=is_dist,
                shuffle=False,
                persistent_workers=True)
        dataloader_args = {
            **dataloader_default_args,
            **data_cfg['loader']
        }
        dataloader_args['shuffle'] = False
        # dataloader，使用常规采样器，不考虑dataset.flag设置（同等对待所有样本）
        self.dataloader = build_dataloader(dataset, **dataloader_args)

        # 每次采样样本数
        self.sampling_num = sampling_num
        if test_fn is None:
            from antgo.framework.helper.runner.test import single_gpu_test
            self.test_fn = single_gpu_test
        else:
            self.test_fn = test_fn    

        # 采样函数
        self.sampling_fn = build_from_cfg(sampling_fn, UNCERTAINTY_SAMPLING)
        self.sampling_count = sampling_count
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir

    def after_run(self, runner):
        # 根据当前模型，挑选出需要标注的样本
        BaseModule.base_stage = 'activelearning'
        sample_results = None
        now_time = time.time()
        print(f'start analysis next round {now_time} waiting samples')
        for _ in range(self.sampling_count):
            # results is dict
            results = self.test_fn(runner.model, self.dataloader)
            if sample_results is None:
                sample_results = [{} for i in range(len(results))]
                for i in range(len(results)):
                    for k,v in results[i].items():
                        sample_results[i][k] = [v]
            else:
                for i in range(len(results)):
                    for k,v in results[i].items():
                        sample_results[i][k].append(v) 
        
        sampling_list, _ = self.sampling_fn(sample_results, self.dataloader.dataset, self.sampling_num)
        # 将采样列表保存成文件，存储到指定目录
        with open(os.path.join(self.out_dir, f'waiting_sample_{now_time}.json'), 'w') as fp:
            json.dump(sampling_list, fp)

        BaseModule.base_stage = ''


@HOOKS.register_module()
class DistActiveLearningHook(ActiveLearningHook):
    def __init__(
        self, 
        data_cfg, 
        sampling_num, 
        out_dir='./activelearning/', 
        test_fn=None, 
        sampling_fn=None, 
        sampling_count=10,
        broadcast_bn_buffer=True, 
        tmpdir=None) -> None:
        if test_fn is None:
            from antgo.framework.helper.runner.test import multi_gpu_test
            test_fn = multi_gpu_test
        
        assert(sampling_fn is not None)
        super().__init__(data_cfg, sampling_num, out_dir, test_fn, sampling_fn, sampling_count, is_dist=True)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir        

    def after_run(self, runner):
        # # 根据当前模型，挑选出需要标注的样本
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = os.path.join(runner.work_dir, '.activelearning_hook')

        now_time = time.time()
        if runner.rank == 0:
            print(f'start analysis next round {now_time} waiting samples')

        # 所有节点都需要运行
        BaseModule.base_stage = 'activelearning'

        sample_results = None
        for _ in range(self.sampling_count):
            # results is dict
            results = self.test_fn(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=False)
            
            if runner.rank == 0:
                # 仅在master节点，合并所有结果
                if sample_results is None:
                    sample_results = [{} for i in range(len(results))]
                    for i in range(len(results)):
                        for k,v in results[i].items():
                            sample_results[i][k] = [v]
                else:
                    for i in range(len(results)):
                        for k,v in results[i].items():
                            sample_results[i][k].append(v) 
            
        BaseModule.base_stage = ''
        if runner.rank == 0:
            # 仅在master节点运行分析样本
            # 根据当前模型，挑选出需要标注的样本
            sampling_list, _ = self.sampling_fn(sample_results, self.dataloader.dataset, self.sampling_num)
            # 将采样列表保存成文件，存储到指定目录
            with open(os.path.join(self.out_dir, f'waiting_sample_{now_time}.json'), 'w') as fp:
                json.dump(sampling_list, fp)
