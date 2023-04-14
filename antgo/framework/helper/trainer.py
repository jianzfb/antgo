from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.framework.helper.models.proxy_module import ProxyModule
from antgo.framework.helper.runner.test import single_gpu_test

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader, dataloader
import torch.optim
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.apis.train import *
from antgo.framework.helper.dataset import (build_dataset, build_dataloader, build_kv_dataloader, build_iter_dataloader)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import *
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.utils import *
from antgo.framework.helper.models.dummy_module import *
from antgo.framework.helper.models.proxy_module import *
from antgo.framework.helper.models.distillation import *
from antgo.framework.helper.runner.hooks.hook import *
from antgo.framework.helper.parallel.utils import is_module_wrapper
from .base_trainer import *
from thop import profile
import copy
import logging
import tempfile

import torch.distributed as dist
from contextlib import contextmanager
import json

try:
    from aimet_common.defs import QuantScheme
    from aimet_torch import bias_correction
    from aimet_torch.quantsim import QuantParams, QuantizationSimModel
    from aimet_torch.cross_layer_equalization import equalize_model
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.onnx_utils import OnnxExportApiArgs
    from aimet_torch.model_preparer import prepare_model
    from aimet_common.utils import start_bokeh_server_session
except:
    print('Dont support aimet.')
    pass


def _default_forward_pass_callback(model, data_loader):
    # switch to eval state
    model.eval()

    iteration = 10
    prog_bar = ProgressBar(iteration)
    count = 0
    for data in data_loader:
        data = data.to('cuda')
        with torch.no_grad():
            model(data)

        prog_bar.update()
        count += 1
        if count >= iteration:
            break

def apply_cross_layer_equalization(model, input_shape):
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        Batch Norm Folding
        Cross Layer Scaling
        High Bias Fold
    Converts any ReLU6 into ReLU.
    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """

    equalize_model(model, input_shape)


def apply_bias_correction(model, data_loader):
    """
    Applies Bias-Correction on the model.
    :param model: The model to quantize
    :param dataloader: DataLoader used during quantization
    :param logdir: Log directory used for storing log files
    :return: None
    """
    # Rounding mode can be 'nearest' or 'stochastic'
    rounding_mode = 'nearest'

    # Number of samples used during quantization
    num_quant_samples = 256*20

    # Number of samples used for bias correction
    num_bias_correct_samples = 256*20

    params = QuantParams(weight_bw=8, act_bw=8, round_mode=rounding_mode, quant_scheme='tf_enhanced')

    # Perform Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=num_quant_samples,
                                 data_loader=data_loader, num_bias_correct_samples=num_bias_correct_samples)


def calculate_quantsim(model, val_dataloader, dummy_input, use_cuda, path, prefix):
    """
    Calculates model accuracy on quantized simulator and returns quantized model with accuracy.
    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param iterations: No of batches to use in computing encodings.
                       Not used in image net dataset
    :param num_val_samples_per_class: No of samples to use from every class in
                                      computing encodings. Not used in pascal voc
                                      dataset
    :param use_cuda: the cuda device.
    :return: a tuple of quantsim and accuracy of model on this quantsim
    """
    # Number of batches to use for computing encodings
    # Only 5 batches are used here to speed up the process, also the
    # number of images in these 5 batches should be sufficient for
    # compute encodings
    quantsim = QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=_default_forward_pass_callback,
                               forward_pass_callback_args=val_dataloader)

    if not os.path.exists(path):
        os.makedirs(path)
    quantsim.export(path=path, filename_prefix=prefix, dummy_input=dummy_input.cpu())
    return quantsim

class Trainer(BaseTrainer):
    def __init__(self, cfg, work_dir="./", gpu_id=-1, distributed=False, diff_seed=True, deterministic=True, find_unused_parameters=False):
        if isinstance(cfg, dict):
            self.cfg = Config.fromstring(json.dumps(cfg), '.json')
        else:
            self.cfg = cfg
            
        self.data_loaders = None
        self.runner = None
        self.work_dir = work_dir
        self.train_generator = None
        self.val_dataloader = None
        self.distributed = distributed
        self.find_unused_parameters = find_unused_parameters
        self.meta = {}

        # set multi-process settings
        setup_multi_processes(self.cfg)

        gpu_id = int(gpu_id)
        device = 'cpu' if gpu_id < 0 else 'cuda'
        self.cfg.gpu_ids = [gpu_id] if gpu_id >= 0 else []
        if self.distributed:
            init_dist(**self.cfg.get('dist_params', {}))
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)
        
        # set random seeds
        seed = init_random_seed(self.cfg.get('seed', 0), device=device)
        seed = seed + dist.get_rank() if diff_seed else seed
        set_random_seed(seed, deterministic=deterministic)
        self.cfg.seed = seed
        self.meta['seed'] = seed
        self.device = device

    def config_dataloader(self, with_validate=False):
        # 创建数据集
        dataset = build_dataset(self.cfg.data.train)

        # 创建数据集加载器
        train_dataloader_default_args = dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            # `num_gpus` will be ignored if distributed
            num_gpus=1,
            dist=self.distributed,
            seed=self.cfg.seed,
            runner_type='EpochBasedRunner',
            persistent_workers=True)

        train_loader_cfg = {
            **train_dataloader_default_args,
            **self.cfg.data.get('train_dataloader', {})
        }

        if getattr(dataset, 'is_kv', False):
            self.train_generator = build_kv_dataloader(dataset, **train_loader_cfg)
        elif isinstance(dataset, torch.utils.data.IterableDataset):
            self.train_generator = build_iter_dataloader(dataset, **train_loader_cfg)
        else:
            self.train_generator = build_dataloader(dataset, **train_loader_cfg)

        if with_validate:
            val_dataloader_default_args = dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                dist=self.distributed,
                shuffle=False,
                persistent_workers=True)
            
            val_dataloader_args = {
                    **val_dataloader_default_args,
                    **self.cfg.data.get('val_dataloader', {})
            }
            val_dataset = build_dataset(self.cfg.data.val)

            if getattr(dataset, 'is_kv', False):
                self.val_dataloader = build_kv_dataloader(val_dataset, **val_dataloader_args)
            elif isinstance(dataset, torch.utils.data.IterableDataset):
                val_dataset.is_infinite = False
                self.val_dataloader = build_iter_dataloader(val_dataset, **val_dataloader_args)
            else:
                self.val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

    def config_model(self, model_builder=None, resume_from=None, load_from=None, revise_keys=[(r'^module\.', '')]):
        # prepare network
        logger = get_logger('model', log_level=self.cfg.get('log_level', logging.INFO))
        logger.info("Creating graph and optimizer...")

        # 构建网络
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)

        # 模型初始化
        model.init_weights()

        if self.distributed:
            if self.cfg.get('syncBN', True):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)

            model = build_ddp(
                model,
                self.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=self.find_unused_parameters)
        else:
            model = build_dp(model, self.device, device_ids=self.cfg.gpu_ids)

        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        # build optimizer
        # 如果构造optimizer调度，则不在全局hook中注册
        optimizer = None
        if 'type' in self.cfg.optimizer:
            # optimizer 简单，全局控制
            optimizer = build_optimizer(model, self.cfg.optimizer)
        else:
            # optimizer 复合，对多个子模块独立控制
            # 需要模型自己控制
            optimizer = {}
            if is_module_wrapper(model):
                model = model.module

            for submodule_name, optimizer_dict in self.cfg.optimizer.items():
                optimizer[submodule_name] = build_optimizer(getattr(model, submodule_name), optimizer_dict)

        # build lr scheduler
        # 如果构造复合lr调度，则不在全局hook中注册
        lr_scheduler = {}
        if 'policy' not in self.cfg.lr_config:
            # lr schedule 复合，对多个子模块独立控制
            for submodule_name, lr_config in self.cfg.lr_config.items():
                assert('policy' in lr_config)

                policy_type = lr_config.pop('policy')
                # If the type of policy is all in lower case, e.g., 'cyclic',
                # then its first letter will be capitalized, e.g., to be 'Cyclic'.
                # This is for the convenient usage of Lr updater.
                # Since this is not applicable for `
                # CosineAnnealingLrUpdater`,
                # the string will not be changed if it contains capital letters.
                if policy_type == policy_type.lower():
                    policy_type = policy_type.title()
                hook_type = policy_type + 'LrUpdaterHook'
                lr_config['type'] = hook_type
                lr_scheduler[submodule_name] = build_from_cfg(lr_config, HOOKS)         

        # build training strategy
        custom_runner = self.cfg.get('runner', dict(type='EpochBasedRunner', max_epochs=1))
        self.runner = build_runner(
            custom_runner,        # 忽略max_epochs，开发者控制最大epoch
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=self.work_dir,
                logger=logger,
                meta=self.meta,
                lr_scheduler=lr_scheduler))

        # an ugly workaround to make .log and .log.json filenames the same
        self.runner.timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())

        optimizer_config = None
        if not isinstance(optimizer, dict):
            # 对于复合optimizer，不进行后台hook
            # 对于optimizer的控制由模型的train_step内部自己控制
            optimizer_config = self.cfg.get('optimizer_config', {})
            if 'type' not in optimizer_config:
                optimizer_config = OptimizerHook(**optimizer_config)

        custom_hooks = self.cfg.get('custom_hooks', None)        
        self.runner.register_training_hooks(
            self.cfg.lr_config if len(lr_scheduler) == 0 else None,     # 学习率调整策略，比如step,warmup等
            optimizer_config,                                           # 优化器的相关后处理，比如限制梯度操作等
            self.cfg.checkpoint_config,                                 # checkpoint相关处理
            self.cfg.log_config,                                         
            self.cfg.get('momentum_config', None),                       
            custom_hooks_config=custom_hooks)

        if self.distributed:
            if isinstance(self.runner, EpochBasedRunner):
                self.runner.register_hook(DistSamplerSeedHook())

        if self.val_dataloader is not None:
            eval_cfg = self.cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = True
            if 'interval' not in eval_cfg:
                eval_cfg['interval'] = 1

            if 'metric' in eval_cfg:
                metric = build_measure(eval_cfg['metric'])
                eval_cfg['metric'] = metric

            if 'type' not in eval_cfg:
                eval_hook = DistEvalHook if self.distributed else EvalHook
                self.runner.register_hook(
                    eval_hook(self.val_dataloader, **eval_cfg), priority='LOW')
            else:
                eval_cfg['dataloader'] = self.val_dataloader
                self.runner.register_hook(
                    build_from_cfg(eval_cfg, HOOKS), priority='LOW'
                )

        if resume_from:
            self.runner.resume(resume_from)
        elif load_from:
            self.runner.load_checkpoint(load_from, revise_keys=revise_keys)
        else:
            if self.distributed:
                # 为了稳妥起见，把rank0的权重保存下来，其余进程进行加载保证所有卡的起始权重相同
                checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
                rank, _ = get_dist_info()
                if rank == 0:
                    torch.save(model.state_dict(), checkpoint_path)

                dist.barrier()
                self.runner.load_checkpoint(checkpoint_path)
        
    def apply_ptq_quant(self, dummy_input, checkpoint, model_builder=None, path='./', prefix='quant'):
        ###############################     STEP - 0    ###############################
        # 计算浮点模型
        logger = get_logger('qat', log_level=self.cfg.log_level)
        if model_builder is not None:
            f32_model = model_builder()
        else:
            f32_model = build_model(self.cfg.model)

        assert(dummy_input is not None)
        # 加载checkpoint
        ckpt = torch.load(checkpoint, map_location='cpu')
        f32_model.load_state_dict(ckpt['state_dict'], strict=True)

        # 计算浮点模型精度
        print('Computing Float Model Accuray.')
        eval_cfg = self.cfg.get('evaluation', {})
        metric_func = None
        if 'metric' in eval_cfg:
            metric_func = build_measure(eval_cfg['metric'])

        warp_model = build_dp(f32_model, self.device, device_ids=self.cfg.gpu_ids)  
        results = single_gpu_test(warp_model, self.val_dataloader)    
        eval_res = {}
        if metric_func is None:
            eval_res = self.val_dataloader.dataset.evaluate(results, logger=None)
        else:
            gts = []
            for gt_i in range(len(self.val_dataloader.dataset)):
                gts.append(self.val_dataloader.dataset.get_ann_info(gt_i))

            eval_res = metric_func(results, gts)                 
        print(eval_res)

        # 获得浮点模型的 FLOPS、PARAMS
        model = DummyModelWarp(f32_model)        
        model = model.to('cpu')
        dummy_input = dummy_input.to('cpu')
        flops, params = profile(model, inputs=(dummy_input,))
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        if not os.path.exists(os.path.join(path, 'float')):
            os.makedirs(os.path.join(path, 'float'))
        # Export the model
        torch.onnx.export(
                model,                                      # model being run
                dummy_input,                                # model input (or a tuple for multiple inputs)
                os.path.join(path, 'float', 'model.onnx'),  # where to save the model (can be a file or file-like object)
                export_params=True,                         # store the trained parameter weights inside the model file
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
        )

        # Quantize the model using AIMET QAT (quantization aware training) and calculate accuracy on Quant Simulator
        ###############################     STEP - 1    ###############################
        print('Computing Naive Quantizing Progress.')
        model = model.to(self.device)
        quant_val_loader =  \
            torch.utils.data.DataLoader(UnlabeledDatasetWrapper(self.val_dataloader.dataset, 'image'),
                                        batch_size=128, 
                                        shuffle=True)     
        dummy_input = dummy_input.to(self.device)
        quant_sim = calculate_quantsim(model, quant_val_loader, dummy_input, True, os.path.join(path, 'NAIVE'), prefix)

        print("Naive Quantized Model performance")
        # self.cfg.qat.update(dict(feature_func=quant_sim.model))
        # warp_model = build_model(self.cfg.qat) 
        head_cfg = None
        for k in self.cfg.model.keys():
            if 'head' in k:
                head_cfg = self.cfg.model[k]
        assert(head_cfg is not None)
        proxy_model_cfg = self.cfg['proxy_model'] if 'proxy_model' in self.cfg.keys() else {}
        warp_model = ProxyModule(quant_sim.model, head_cfg, init_cfg=proxy_model_cfg['init_cfg'] if 'init_cfg' in proxy_model_cfg else None)
        warp_model = build_dp(warp_model, self.device, device_ids=self.cfg.gpu_ids)        
        results = single_gpu_test(warp_model, self.val_dataloader)            
        eval_res = {}
        if metric_func is None:
            eval_res = self.val_dataloader.dataset.evaluate(results, logger=None)
        else:
            gts = []
            for gt_i in range(len(self.val_dataloader.dataset)):
                gts.append(self.val_dataloader.dataset.get_ann_info(gt_i))

            eval_res = metric_func(results, gts)                 
        print(eval_res)        
        
        ###############################     STEP - 2    ###############################
        # For good initialization apply, apply Post Training Quantization (PTQ) methods
        # such as Cross Layer Equalization (CLE) and Bias Correction (BC) (optional)
        print('Computing PTQ(CLE and BC) Quantizing Progress.')
        apply_cross_layer_equalization(model=model, input_shape=dummy_input.shape)
        apply_bias_correction(model=model, 
                              data_loader=torch.utils.data.DataLoader(LabeledDatasetWrapper(self.val_dataloader.dataset, 'image'), batch_size=256, shuffle=True))
        dummy_input = dummy_input.to(self.device)
        quant_sim = calculate_quantsim(model, quant_val_loader, dummy_input, True, os.path.join(path, 'PTQ'), prefix)
        
        print('Post Training Quantization (PTQ) Complete')
        # self.cfg.qat.update(dict(feature_func=quant_sim.model))
        # warp_model = build_model(self.cfg.qat) 
        warp_model = ProxyModule(quant_sim.model, head_cfg, init_cfg=proxy_model_cfg['init_cfg'] if 'init_cfg' in proxy_model_cfg else None)
        warp_model = build_dp(warp_model, self.device, device_ids=self.cfg.gpu_ids)
        results = single_gpu_test(warp_model, self.val_dataloader)  
        eval_res = {}
        if metric_func is None:
            eval_res = self.val_dataloader.dataset.evaluate(results, logger=None)
        else:
            gts = []
            for gt_i in range(len(self.val_dataloader.dataset)):
                gts.append(self.val_dataloader.dataset.get_ann_info(gt_i))

            eval_res = metric_func(results, gts)                 
        print(eval_res)           

    def apply_qat_quant(self, dummy_input, checkpoint, model_builder=None, path='./', prefix='quant'):
        logger = get_logger('qat', log_level=self.cfg.log_level)
        if model_builder is not None:
            f32_model = model_builder()
        else:
            f32_model = build_model(self.cfg.model)

        assert(dummy_input is not None)
        # 加载checkpoint
        ckpt = torch.load(checkpoint, map_location='cpu')
        f32_model.load_state_dict(ckpt['state_dict'], strict=True)

        model = DummyModelWarp(f32_model)        
        model = model.to(self.device)
        quant_val_loader =  \
            torch.utils.data.DataLoader(
                UnlabeledDatasetWrapper(self.val_dataloader.dataset, 'image'),batch_size=128, 
                                        shuffle=False)    

        ###############################     STEP - 0    ###############################
        # For good initialization apply, apply Post Training Quantization (PTQ) methods
        # such as Cross Layer Equalization (CLE) and Bias Correction (BC) (optional)
        if not self.distributed or (get_dist_info()[0] == 0):
            print('Computing PTQ(CLE and BC) Quantizing Progress.')
        apply_cross_layer_equalization(model=model, input_shape=dummy_input.shape)
        apply_bias_correction(model=model, 
                              data_loader=torch.utils.data.DataLoader(
                                  LabeledDatasetWrapper(self.val_dataloader.dataset, 'image'), 
                                  batch_size=256, 
                                  shuffle=True))
        dummy_input = dummy_input.to(self.device)
        quant_sim = calculate_quantsim(model, quant_val_loader, dummy_input, True, os.path.join(path, 'PTQ'), prefix)

        # 删除原始模型      
        del model

        if not self.distributed or (get_dist_info()[0] == 0):
            print('Post Training Quantization (PTQ) Complete')
        head_cfg = None
        for k in self.cfg.model.keys():
            if 'head' in k:
                head_cfg = self.cfg.model[k]
        assert(head_cfg is not None)            
        proxy_model_cfg = self.cfg['proxy_model'] if 'proxy_model' in self.cfg.keys() else {}        
        model = ProxyModule(quant_sim.model, head_cfg, init_cfg=proxy_model_cfg['init_cfg'] if 'init_cfg' in proxy_model_cfg else None)
        
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = build_ddp(
                model,
                self.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = build_dp(model, self.device, device_ids=self.cfg.gpu_ids)

        ###############################     STEP - 1    ###############################
        # Finetune the quantized model
        if not self.distributed or (get_dist_info()[0] == 0):        
            print('Starting Model Finetune')
        # build optimizer        
        optimizer = build_optimizer(model, self.cfg.optimizer)

        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        # build training strategy
        self.runner = build_runner(
            dict(type='EpochBasedRunner', max_epochs=1),        # 忽略max_epochs，开发者控制最大epoch
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=self.work_dir,
                logger=logger,
                meta=self.meta))

        # an ugly workaround to make .log and .log.json filenames the same
        self.runner.timestamp = time.strftime('%Y-%m-%dx%H-%M-%S', time.localtime())

        optimizer_config = self.cfg.optimizer_config
        if self.distributed and 'type' not in self.cfg.optimizer_config:
            optimizer_config = OptimizerHook(**self.cfg.optimizer_config)

        self.runner.register_training_hooks(
            self.cfg.lr_config,                                     # 学习率调整策略，比如step,warmup等
            optimizer_config,                                       # 优化器的相关后处理，比如限制梯度操作等
            self.cfg.checkpoint_config,                             # checkpoint相关处理
            self.cfg.log_config,                                         
            self.cfg.get('momentum_config', None),                       
            custom_hooks_config=self.cfg.get('custom_hooks', None))

        if self.distributed:
            if isinstance(self.runner, EpochBasedRunner):
                self.runner.register_hook(DistSamplerSeedHook())

        # 配置验证方法
        eval_cfg = self.cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = True
        if 'interval' not in eval_cfg:
            eval_cfg['interval'] = 1

        eval_hook = DistEvalHook if self.distributed else EvalHook

        if 'metric' in eval_cfg:
            metric = build_measure(eval_cfg['metric'])
            eval_cfg['metric'] = metric
        self.runner.register_hook(
            eval_hook(self.val_dataloader, **eval_cfg), priority='LOW')

        # 训练
        with self.train_context(15) as runner:
            for _ in range(0, runner._max_epochs):
                self.run_on_train()

        # Save the quantized model
        # only on rank=0 node
        if not self.distributed or (get_dist_info()[0] == 0):
            if not os.path.exists(os.path.join(path, 'QAT')):
                os.makedirs(os.path.join(path, 'QAT'))
            quant_sim.export(path=os.path.join(path, 'QAT'), filename_prefix=prefix, dummy_input=dummy_input.cpu())