from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from fileinput import filename

import os
import os.path as osp
import math
import time
import glob
import abc
from antgo.framework.helper.runner.hooks import HOOKS, Hook
from antgo.framework.helper.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import (build_dataloader,build_kv_dataloader, build_dataset)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.models.builder import *
import torch.distributed as dist
from contextlib import contextmanager
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.runner.test import multi_gpu_test, single_gpu_test
from antgo.framework.helper.cnn.utils import fuse_conv_bn
from antgo.framework.helper.tester import Tester
from antgo.framework.helper.runner.hooks.samplingmethods.kcenter_greedy import *
from antgo.framework.helper.runner import BaseModule
from thop import profile
from antgo.utils.sample_gt import *
from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper.task_flag import *
import json
import imagesize
from antgo.framework.helper.fileio import *


UNCERTAINTY_SAMPLING = Registry('UNCERTAINTY_SAMPLING')

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)


@UNCERTAINTY_SAMPLING.register_module()
class SamplingByEntropy(object):
    def __init__(self, key, sampling_num=0, use_softmax=True, ratio=0.1) -> None:
        super().__init__()
        self.key = key
        self.sampling_num = sampling_num
        self.use_softmax = use_softmax
        self.ratio = ratio
    
    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
        entropy_list = []
        if pre_selected is None:
            pre_selected = list(range(len(results))) 
        
        if self.sampling_num > 0:
            sampling_num = self.sampling_num
        for i in pre_selected:
            # hand_class_preds: NxBx4
            hand_class_preds = np.stack(results[i][self.key], 0)
            # multi-round mean
            hand_class_preds = np.mean(hand_class_preds, 0)            
            if self.use_softmax:
                hand_class_preds = softmax(hand_class_preds, -1)

            C = hand_class_preds.shape[-1]
            hand_class_preds = np.reshape(hand_class_preds, (-1, C))
            entropy_val_list = []
            for j in range(hand_class_preds.shape[0]):
                entropy_val_list.append(
                    entropy(hand_class_preds[j])
                )
     
            if self.ratio > 0:
                entropy_val_list = sorted(entropy_val_list, reverse=True)                       
                use_num = (int)(self.ratio * len(entropy_val_list))
                entropy_val_list = entropy_val_list[:use_num]
            entropy_list.append((np.mean(entropy_val_list), i))

        ordered_unlabeled = sorted(entropy_list, key=lambda x: x[0], reverse=True)
        ordered_unlabeled = ordered_unlabeled[:sampling_num]
        ordered_indexs = [s[1] for s in ordered_unlabeled]
        
        sample_list = []
        selected_list = []
        for index in ordered_indexs:
            # get index in dataset
            selected_list.append(index)

            meta_info = results[index]['image_meta'][0]
            image_file = meta_info['image_file']
            image_index = meta_info.get('id', '')
            sample_list.append({
                'id': image_index,
                'image_file': image_file,
                'tag': meta_info['tag']
            })

        return sample_list, selected_list
        

@UNCERTAINTY_SAMPLING.register_module()
class SamplingByUniform(object):
    def __init__(self, sampling_num=0) -> None:
        super().__init__()
        self.sampling_num = sampling_num
    
    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
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

            meta_info = results[index]['image_meta'][0]
            image_file = meta_info['image_file']
            image_index = meta_info.get('id', '')
            sample_list.append({
                'id': image_index,
                'image_file': image_file,
                'tag': meta_info['tag']
            })

        return sample_list, selected_list


@UNCERTAINTY_SAMPLING.register_module()
class SamplingByKCenter(object):
    def __init__(self, key, sampling_num=0) -> None:
        super().__init__()
        self.key = key
        self.sampling_num = sampling_num
    
    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
        # 1.step stack feature
        feature_stack = []
        if pre_selected is None:
            pre_selected = list(range(len(results)))        
        for i in pre_selected:
            feature = results[i][self.key]
            if type(feature) == list:
                feature = np.mean(np.stack(feature, 0),0)

            feature_stack.append(feature.flatten())
        # NxC
        feature_stack = np.stack(feature_stack, 0)

        # 2.step centergreedy
        if self.sampling_num > 0:
            sampling_num = self.sampling_num        
        kcenter_greedy = kCenterGreedy(feature_stack)
        selected_i_list = kcenter_greedy.select_batch(already_selected=[], N=sampling_num)

        # 3.step select sample
        sample_list = []
        selected_list = []
        for index in selected_i_list:
            # get index in dataset
            index = pre_selected[index]
            selected_list.append(index)    

            meta_info = results[index]['image_meta'][0]
            image_file = meta_info['image_file']
            image_index = meta_info.get('id', '')
            sample_list.append({
                'id': image_index,
                'image_file': image_file,
                'tag': meta_info['tag']
            })

        return sample_list, selected_list


@UNCERTAINTY_SAMPLING.register_module()
class GuidByCrossDomain(object):
    def __init__(self, key, sampling_num=0) -> None:
        super().__init__()
        self.key = key
        self.sampling_num = sampling_num
            
    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
        # 1.step stack feature
        feature_stack = []
        if pre_selected is None:
            pre_selected = list(range(len(results)))        
        for i in pre_selected:
            feature = results[i][self.key]
            if type(feature) == list:
                feature = np.mean(np.stack(feature, 0),0)

            feature_stack.append(feature.flatten())
        # NxC
        feature_stack = np.stack(feature_stack, 0)
        N = feature_stack.shape[0]

        # 2.step stack ref feature
        ref_feature_stack = []
        for i in range(len(ref_results)):
            ref_feature = ref_results[i][self.key]
            if type(ref_feature) == list:
                ref_feature = np.mean(np.stack(ref_feature, 0),0)

            ref_feature_stack.append(ref_feature.flatten())
        # NxC
        ref_feature_stack = np.stack(ref_feature_stack, 0)

        # 3.step mining
        if self.sampling_num > 0:
            sampling_num = self.sampling_num        
        
        min_distances = None
        new_batch = []
        ind = 0
        inf_distances = np.zeros((feature_stack.shape[0], 1))
        ref_ind_list = []
        for _ in range(sampling_num):
            if min_distances is None:
                ind = np.random.choice(np.arange(N))
                inf_distances[ind] = 10000000.0
                min_distances = pairwise_distances(ref_feature_stack, feature_stack[ind:ind+1])
            else:
                ref_ind = np.argmax(min_distances)
                temp = pairwise_distances(feature_stack, ref_feature_stack[ref_ind:ref_ind+1])
                temp = np.maximum(inf_distances, temp)
                ind = np.argmin(temp)
                inf_distances[ind] = 10000000.0
            
                # 计算训练集中的ind样本，与参考集中的样本的最小距离，并更新
                min_distances_new = pairwise_distances(ref_feature_stack, feature_stack[ind:ind+1])
                min_distances = np.minimum(min_distances, min_distances_new)
                min_distances[ref_ind] = 0.0
                ref_ind_list.append(ref_ind)
            
            new_batch.append(ind)

        # 4.step select sample
        sample_list = []
        selected_list = []
        for index in new_batch:
            # get index in dataset
            index = pre_selected[index]
            selected_list.append(index)

            meta_info = results[index]['image_meta'][0]
            image_file = meta_info['image_file']
            image_index = meta_info.get('id', '')
            sample_list.append({
                'id': image_index,
                'image_file': image_file,
                'tag': meta_info['tag']
            })

        return sample_list, selected_list


@UNCERTAINTY_SAMPLING.register_module()
class SmplingByMix(object):
    def __init__(self, sampling_cfgs):
        super().__init__()
        self.sampling_fns= []
        for cfg in sampling_cfgs:
            sampling_fn = build_from_cfg(cfg, UNCERTAINTY_SAMPLING)
            self.sampling_fns.append(sampling_fn)        

    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
        mix_sample_list = []
        mix_selected_list = []

        if pre_selected is None:
            pre_selected = list(range(len(results)))    

        for sampling_fn in self.sampling_fns:
            sample_list, selected_list = sampling_fn(results, sampling_num, pre_selected, ref_results)

            mix_sample_list.extend(sample_list)
            mix_selected_list.extend(selected_list)

            # 更新pre_selected
            # 从pre_selected 中抛出selected_list
            pre_selected = list(set(pre_selected)-set(selected_list))

        return mix_sample_list, mix_selected_list


@UNCERTAINTY_SAMPLING.register_module()
class SamplingByComposer(object):
    def __init__(self, sampling_cfgs) -> None:
        super().__init__()
        self.sampling_fns= []
        for cfg in sampling_cfgs:
            sampling_fn = build_from_cfg(cfg, UNCERTAINTY_SAMPLING)
            self.sampling_fns.append(sampling_fn)

    def __call__(self, results, sampling_num, pre_selected=None, ref_results=None):
        sample_list = []        
        if pre_selected is None:
            pre_selected = list(range(len(results)))    

        selected_list = pre_selected
        for sampling_fn in self.sampling_fns:
            sample_list, selected_list = sampling_fn(results, sampling_num, selected_list, ref_results)

        return sample_list, selected_list


class Activelearning(Tester):
    def __init__(self, cfg, work_dir='./', gpu_id=-1, distributed=False):
        self.cfg = cfg
        # 默认使用单卡进行主动学习即可
        assert(not distributed)

        # 远程下载历史数据        
        if getattr(cfg, 'root', '') != '':
            file_client_mkdir(self.cfg.root, True)
            dir_name = os.path.dirname(cfg.root)
            if file_client_exists(os.path.join(dir_name, 'activelearning')):
                file_client_get(os.path.join(dir_name, 'activelearning'), './')

        if not os.path.exists('./activelearning'):
            os.mkdir('./activelearning')

        # has finish round info
        has_finish_anns = []
        for sub_folder in os.listdir('./activelearning'):
            if os.path.exists(f'./activelearning/{sub_folder}/annotation.json'):
                has_finish_anns.append(f'./activelearning/{sub_folder}/annotation.json')

        unlabel_filter_warp = dict(
            type='IterableDatasetFilter',
            dataset=build_from_cfg(cfg.data.test, DATASETS, None),
            not_in_anns=has_finish_anns
        )

        cfg.data.test = unlabel_filter_warp
        super().__init__(cfg, work_dir, gpu_id, distributed)

        # 采样函数
        self.sampling_fn = build_from_cfg(cfg.model.init_cfg.sampling_fn, UNCERTAINTY_SAMPLING)
        self.sampling_count = cfg.model.init_cfg.sampling_count # 计算预测次数
        self.sampling_num = cfg.model.init_cfg.sampling_num     # 最终采样数（最有价值的等待标注样本）

    def select(self, exp_name):
        # 添加运行标记
        running_flag(self.cfg.get('root', None))

        rank, _ = get_dist_info()        
        sample_results = None
        for _ in range(self.sampling_count):
            if not self.distributed:
                results = single_gpu_test(self.model, self.data_loader, needed_info=['image_meta'])
            else:
                results = multi_gpu_test(self.model, self.data_loader, needed_info=['image_meta'])

            if rank == 0:
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
                            
            # 同步
            if self.distributed:
                dist.barrier()
        
        if rank == 0:
            # 仅在master节点运行分析样本
            # step1: 根据当前模型，挑选出需要标注的样本
            sampling_list, _ = self.sampling_fn(sample_results, self.sampling_num, ref_results=None)

            # step2: 遍历数据找到挑选出的样本，并保存到目标            
            # step2.1: 清空处理管线
            from antgo.framework.helper.dataset import PIPELINES
            self.dataset.dataset.pipeline = [build_from_cfg(dict(type='Meta', keys=['image_file', 'tag']), PIPELINES)]  # 仅添加Meta处理管线
            self.dataset.dataset._fields = ["image", "image_meta"] #
            self.dataset.dataset._alias = ["image", "image_meta"] #

            # step2.2: 保存到目标目录
            target_folder = f'./activelearning/{exp_name}'
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            image_folder = os.path.join(target_folder, 'images')
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            # TODO，每个样本如何进行唯一标识
            selected_sample_ids = [f"{sample['tag']}/{sample['image_file']}" for sample in sampling_list]

            image_file_name_list = []
            compare_pos = 0
            for sample in self.dataset:
                sample_tag = sample['image_meta']['tag']
                sample_id = f"{sample_tag}/{sample['image_meta']['image_file']}"
                
                if sample_id == selected_sample_ids[compare_pos]:
                    compare_pos += 1
                    # 防止在原目录下image_file是多层级的，在此将/替换为-
                    file_name = f"{sample_tag}_{sample['image_meta']['image_file'].replace('/','-')}"
                    if not (file_name.lower().endswith('.png') or file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg')):
                        file_name = f'{file_name}.png'
                    image_file_name_list.append((file_name, sample_tag))
                    cv2.imwrite(os.path.join(image_folder, file_name), sample['image'])

            # step2.3: 生成解析文件
            # 生成默认格式解析json
            annotation_file_name = 'annotation.json'
            annotation_list = []
            # 加载默认标准格式
            sgt = SampleGTTemplate() 
            for sample_file_name, sample_file_tag in image_file_name_list:
                sample_file_path = os.path.join(image_folder, sample_file_name)
                width, height = imagesize.get(sample_file_path)

                sample_gt = sgt.get()
                sample_gt_cp = copy.deepcopy(sample_gt)
                sample_gt_cp.update({
                    'image_file': f'images/{sample_file_name}'
                })
                sample_gt_cp['tag'] = f'{sample_file_tag}'
                sample_gt_cp['height'] = height
                sample_gt_cp['width'] = width
                annotation_list.append(sample_gt_cp)

            with open(os.path.join(target_folder, annotation_file_name), 'w', encoding="utf-8") as fp:
                json.dump(annotation_list, fp)

            with open(os.path.join(target_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
                meta_info= sgt.meta()
                meta_info['extent']['total'] = len(annotation_list)
                json.dump(meta_info, fp)

            # step2.4: 保存到远程
            if getattr(self.cfg, 'root', '') != '':
                # 记录本次挑选出的数据记录
                dir_name = os.path.dirname(self.cfg.root)
                target_path = os.path.join(dir_name, 'activelearning', exp_name)
                if not file_client_exists(target_path):
                    file_client_mkdir(target_path, True)
                file_client_put(target_path, os.path.join(target_folder, annotation_file_name), False)

                # 打包本次挑选出的数据，并保存到标注目录下
                # root/label/exp_name/data.tar
                target_path = os.path.join(dir_name, 'label', exp_name)
                if not file_client_exists(target_path):
                    file_client_mkdir(target_path, True)

                os.system(f'cd {target_folder} && tar -cf ../data.tar .')
                file_client_put(target_path, './activelearning/data.tar', False)
                os.system('rm ./activelearning/data.tar')

            # 添加完成标记
            finish_flag(self.cfg.get('root', None))