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
from antgo.framework.helper.dataset import (build_dataloader, build_dataset)
from antgo.framework.helper.utils.util_distribution import build_ddp, build_dp, get_device
from antgo.framework.helper.utils import get_logger
from antgo.framework.helper.runner import get_dist_info, init_dist
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.utils.setup_env import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.utils import *
from antgo.framework.helper.models.dummy_module import *
from antgo.framework.helper.models.proxy_module import *
from thop import profile
import copy

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
    from aimet_torch import visualize_model
except:
    print('Dont support aimet.')
    pass

'''
cfg_dict = dict(
    optimizer = dict(type='Adam', lr=1e-4),
    optimizer_config = dict(grad_clip=None),
    lr_config = dict(
        policy='step',
        by_epoch=True,
        warmup=None,
        gamma=0.1,
        step=[15, 17]),
    log_config = dict(
        interval=5,    
        hooks=[
            dict(type='TextLoggerHook'),
        ]),
    checkpoint_config = dict(interval=1,out_dir='./'),        
    seed=0,
    data = dict(
        train=dict(
                    type='InterHand26MReader',
                    dir='/root/paddlejob/workspace/env_run/portrait/InterHand2.6M/data/InterHand2.6M',
                    trans_test='rootnet',
                    output_hm_shape=(64, 64, 64),
                    input_img_shape=(256, 256),
                    bbox_3d_size=400,      
                    output_root_hm_shape=64, 
                    bbox_3d_size_root=400,
                    pipeline=[dict(type='Compose', processes=[['ToTensor', {}]])],
                    inputs_def = {
                        'fields': [
                            ['image'],
                            ['joint_coord','rel_root_depth','hand_type'],
                            ['joint_valid', 'root_valid', 'hand_type_valid', 'inv_trans', 'capture', 'cam', 'frame']
                        ]
                    }
                ),
        train_dataloader=dict(samples_per_gpu=32,workers_per_gpu=2),
        val=dict(
                type='InterHand26MReader',
                dir='/root/paddlejob/workspace/env_run/portrait/InterHand2.6M/data/InterHand2.6M',
                train_or_test='val',
                trans_test='rootnet',
                output_hm_shape=(64, 64, 64),
                input_img_shape=(256, 256),
                bbox_3d_size=400,  
                output_root_hm_shape=64, 
                bbox_3d_size_root=400,
                pipeline=[dict(type='Compose', processes=[['ToTensor', {}]])],
                inputs_def = {
                    'fields': [
                        ['image'],
                        ['joint_coord','rel_root_depth','hand_type'],
                        ['joint_valid', 'root_valid', 'hand_type_valid', 'inv_trans', 'capture', 'cam', 'frame', 'TID']
                    ]
                }
            ),
        val_dataloader=dict(samples_per_gpu=16, workers_per_gpu=1),
        ),
    gpu_ids=[0],
    log_level=logging.INFO,
    evaluation=dict(out_dir='./out')
)
'''
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

class Trainer(object):
    def __init__(self, cfg_dict, work_dir="./", device='cuda', distributed=False, diff_seed=True, deterministic=True):
        self.cfg = Config.fromstring(json.dumps(cfg_dict), '.json')

        self.data_loaders = None
        self.runner = None
        self.work_dir = work_dir
        self.train_generator = None
        self.val_dataloader = None
        self.distributed = distributed
        self.meta = {}

        # set multi-process settings
        # 如 opencv_num_threads, OMP_NUM_THREADS, MKL_NUM_THREADS
        setup_multi_processes(self.cfg)

        if self.distributed:
            init_dist("pytorch", **self.cfg.get('dist_params', {}))
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

    def make_dataloader(self, with_validate=False):
        # 创建数据集
        dataset = build_dataset(self.cfg.data.train)

        # 创建数据集加载器
        train_dataloader_default_args = dict(
            samples_per_gpu=2,
            workers_per_gpu=2,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(self.cfg.gpu_ids),
            dist=self.distributed,
            seed=self.cfg.seed,
            runner_type='EpochBasedRunner',
            persistent_workers=True)

        train_loader_cfg = {
            **train_dataloader_default_args,
            **self.cfg.data.get('train_dataloader', {})
        }
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
            val_dataset = build_dataset(self.cfg.data.val, dict(test_mode=True))
            self.val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)

    def make_model(self, model_builder=None, resume_from=None, load_from=None):
        # prepare network
        logger = get_logger('model', log_level=self.cfg.log_level)
        logger.info("Creating graph and optimizer...")
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)
        
        # 模型初始化
        model.init_weights()

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

        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        # build optimizer
        optimizer = build_optimizer(model, self.cfg.optimizer)

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
        self.runner.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

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

        if self.val_dataloader is not None:
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

        if resume_from is not None:
            self.runner.resume(resume_from)
        elif load_from is not None:
            self.runner.load_checkpoint(load_from)

    @contextmanager
    def train_context(self, max_epochs):
        self.runner._max_epochs = max_epochs
        self.runner.call_hook('before_run')
        yield self.runner

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.runner.call_hook('after_run')

    def run_on_train(self, **kwargs):
        self.runner.train(self.train_generator, **kwargs)

    def run_on_val(self, **kwargs):
        self.runner.val(self.val_dataloader, **kwargs)
    
    def apply_ada_quant(self, dummy_input, checkpoint, model_builder=None, num_batches=4, default_num_iterations=10000, filename_prefix='quant', path='./'):
        assert(self.val_dataloader is not None)
        eval_cfg = self.cfg.get('evaluation', {})
        metric_func = None
        if 'metric' in eval_cfg:
            metric_func = build_measure(eval_cfg['metric'])

        print('load model and checkpoint.')
        if model_builder is not None:
            f32_model = model_builder()
        else:
            f32_model = build_model(self.cfg.model)

        ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
        f32_model.load_state_dict(ckpt['state_dict'], strict=True)

        # 计算浮点模型精度
        print('computing float model accuray.')
        results = single_gpu_test(f32_model, self.val_dataloader)
        eval_res = {}
        if metric_func is None:
            eval_res = self.val_dataloader.dataset.evaluate(results, logger=None)
        else:
            gts = []
            for gt_i in range(len(self.val_dataloader.dataset)):
                gts.append(self.val_dataloader.dataset.get_ann_info(gt_i))

            eval_res = metric_func(results, gts)        
        print(eval_res)

        # 开始量化分析
        print('starting ada process.')
        # fold bn
        model = DummyModelWarp(f32_model)        
        _ = fold_all_batch_norms(model, input_shapes=dummy_input.shape)
        model = model.to(self.device)
        prepared_model = prepare_model(model)
        quant_train_loader = \
            torch.utils.data.DataLoader(UnlabeledDatasetWrapper(self.train_generator.dataset, 'image'),
                                        batch_size=32, 
                                        shuffle=True)
        quant_val_loader =  \
            torch.utils.data.DataLoader(UnlabeledDatasetWrapper(self.val_dataloader.dataset, 'image'),
                                        batch_size=4, 
                                        shuffle=True)

        params = AdaroundParameters(data_loader=quant_train_loader, 
                                    num_batches=32, 
                                    default_num_iterations=default_num_iterations,
                                    default_reg_param=0.01, default_beta_range=(20, 2))

        if not os.path.exists(path):
            os.makedirs(path)
        # Returns model with adarounded weights and their corresponding encodings
        adarounded_model = Adaround.apply_adaround(
                                        prepared_model, dummy_input, params, 
                                        path=path,
                                        filename_prefix=filename_prefix, 
                                        default_param_bw=8,
                                        default_quant_scheme=QuantScheme.post_training_tf_enhanced,
                                        default_config_file=None)

        sim = QuantizationSimModel(adarounded_model, 
                                    quant_scheme=QuantScheme.post_training_tf_enhanced, 
                                    default_param_bw=8,
                                    default_output_bw=8, 
                                    dummy_input=dummy_input)

        # Set and freeze encodings to use same quantization grid and then invoke compute encodings
        sim.set_and_freeze_param_encodings(encoding_path=os.path.join(path, f'{filename_prefix}.encodings'))

        # compute encodings on val
        sim.compute_encodings(_default_forward_pass_callback, 
                              forward_pass_callback_args=quant_val_loader) 

        # 输出量化模型
        print('export quant model.')
        self.export(path, filename_prefix, dummy_input, quant_sim=sim)

        # 计算量化模型精度
        print('computing quant model accuray.')
        f32_model.set_extract_feat_func(sim.model)
        quant_model = build_dp(f32_model, self.device, device_ids=self.cfg.gpu_ids)
        results = single_gpu_test(quant_model, self.val_dataloader)
        eval_res = {}
        if metric_func is None:
            eval_res = self.val_dataloader.dataset.evaluate(results, logger=None)
        else:
            gts = []
            for gt_i in range(len(self.val_dataloader.dataset)):
                gts.append(self.val_dataloader.dataset.get_ann_info(gt_i))

            eval_res = metric_func(results, gts)        
        print(eval_res)

    def export(self, dummy_input, checkpoint=None, model_builder=None, path='./', prefix='model'):
        f32_model = None
        if model_builder is not None:
            f32_model = model_builder()
        else:
            f32_model = build_model(self.cfg.model)

        # 加载checkpoint
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu')
            f32_model.load_state_dict(ckpt['state_dict'], strict=True)

        # 获得浮点模型的 FLOPS、PARAMS
        model = DummyModelWarp(f32_model)        
        model = model.eval()
        model = model.to('cpu')
        dummy_input = dummy_input.to('cpu')
        flops, params = profile(model, inputs=(dummy_input,))
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        if not os.path.exists(path):
            os.makedirs(path)

        # Export the model
        torch.onnx.export(
                model,                                      # model being run
                dummy_input,                                # model input (or a tuple for multiple inputs)
                os.path.join(path, f'{prefix}.onnx'),       # where to save the model (can be a file or file-like object)
                export_params=True,                         # store the trained parameter weights inside the model file
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = ['input'],                    # the model's input names
        )

        # 模型可视化
        if not os.path.exists(os.path.join(path, 'visualization')):
            os.makedirs(os.path.join(path, 'visualization'))
        visualization_dir = os.path.join(path, 'visualization')
        visualize_model.visualize_weight_ranges(model, visualization_dir)
        visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, visualization_dir)

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

        # 自动调整单设备下的学习率到多设备下的学习率
        auto_scale_lr(self.cfg, self.distributed, logger)

        ###############################     STEP - 1    ###############################
        # Finetune the quantized model
        if not self.distributed or (get_dist_info()[0] == 0):        
            print('Starting Model Finetune')
        # build optimizer        
        optimizer = build_optimizer(model, self.cfg.optimizer)

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
        self.runner.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

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