from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.dataset import build_dataset
from antgo.framework.helper.runner.dist_utils import master_only
import antvis.client.mlogger as mlogger
from antgo.utils import *
import numpy as np
import cv2
from thop import profile
import json
from collections import OrderedDict
import re
import shutil


class Exporter(object):
    def __init__(self, cfg, work_dir, **kwargs):
        if isinstance(cfg, dict):
            self.cfg = Config.fromstring(json.dumps(cfg), '.json')
        else:
            self.cfg = cfg
        self.work_dir = work_dir
        self.use_logger_platform = False

    @master_only
    def _finding_from_logger(self, experiment_name, checkpoint_name):
        # step 1: 检测当前路径下收否有token缓存
        token = None
        if os.path.exists('./.token'):
            with open('./.token', 'r') as fp:
                token = fp.readline()

        # step 2: 检查antgo配置目录下的配置文件中是否有token
        if token is None or token == '':
            config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
            config.AntConfig.parse_xml(config_xml)
            token = getattr(config.AntConfig, 'server_user_token', '')
        if token == '' or token is None:
            print('No valid vibstring token, directly return')
            return None, None

        # 创建实验
        mlogger.config(token=token)
        project_name = self.cfg.get('project_name', os.path.abspath(os.path.curdir).split('/')[-1])
        status = mlogger.activate(project_name, experiment_name)
        if status is None:
            print(f'Couldnt find {project_name}/{experiment_name}, from logger platform')
            return None, None

        file_logger = mlogger.Container()
        local_config_path = None
        local_checkpoint_path = None
        remote_checkpoint_path = None
        # 下载配置文件
        mlogger.FileLogger.cache_folder = f'./logger/cache/{experiment_name}'
        file_logger.cfg_file = mlogger.FileLogger('config', 'qiniu')
        file_list, _ = file_logger.cfg_file.get()
        if len(file_list) > 0:
           local_config_path = file_list[0]

        # 下载checkpoint文件
        file_logger.checkpoint_file = mlogger.FileLogger('file', 'aliyun')
        file_list, remote_list = file_logger.checkpoint_file.get(checkpoint_name)

        for file_name, remote_info in zip(file_list, remote_list):
            if file_name.endswith(checkpoint_name):
                local_checkpoint_path = file_name
                remote_checkpoint_path = remote_info
                break
        print(f'Found {local_config_path} {local_checkpoint_path}')
        self.use_logger_platform = True
        remote_checkpoint_path = '/'.join(remote_checkpoint_path.split('/')[:-2])
        mlogger.FileLogger.root_folder = f'{remote_checkpoint_path}/export'
        return local_config_path, local_checkpoint_path

    def export(self, input_tensor_list, input_name_list, output_name_list=None, checkpoint=None, model_builder=None, prefix='model', opset_version=12, revise_keys=[], strict=True, is_dynamic=False, skip_flops_stats=False):
        # 构建模型
        model = None
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)

        # 发现checkpoint
        if checkpoint is None or checkpoint == '':
            checkpoint = self.cfg.get('checkpoint', checkpoint)

        # checkpoint路径格式
        # 1: local path                 本地目录
        # 2: ali://                     直接从阿里云盘下载
        # 3: experiment/checkpoint      日志平台（推荐）
        if checkpoint is not None:
            if not os.path.exists(checkpoint) and len(checkpoint[1:].split('/')) == 2:
                # 尝试解析来自于日志平台
                self.experiment_name, self.checkpoint_name = checkpoint[1:].split('/')
                _, checkpoint = self._finding_from_logger(self.experiment_name, self.checkpoint_name)

        if checkpoint is None or checkpoint == '':
            logger.error('Checkpoint not found')
            return

        # 加载checkpoint
        ckpt = torch.load(checkpoint, map_location='cpu')
        state_dict = ckpt
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                for k, v in state_dict.items()})
        model.load_state_dict(state_dict, strict=strict)

        # 获得浮点模型的 FLOPS、PARAMS
        model.eval()
        model.switch_to_deploy(self.cfg.model.get('test_cfg', dict()))
        model.forward = model.onnx_export
        model = model.to('cpu')
        if isinstance(input_tensor_list, list):
            for i in range(len(input_tensor_list)):
                input_tensor_list[i] = input_tensor_list[i].to('cpu')

        if not skip_flops_stats:
            flops, params = profile(model, inputs=input_tensor_list)
            print('FLOPs = ' + str(flops/1000**3) + 'G')
            print('Params = ' + str(params/1000**2) + 'M')

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # Export the model
        dynamic_axes = None
        if is_dynamic:
            dynamic_axes = {}
            for input_node_name in input_name_list:
                dynamic_axes[input_node_name] = [0]
            for output_node_name in output_name_list:
                dynamic_axes[output_node_name] = [0]

        torch.onnx.export(
                model,                                      # model being run
                tuple(input_tensor_list),                   # model input (or a tuple for multiple inputs)
                os.path.join(self.work_dir, f'{prefix}.onnx'),       # where to save the model (can be a file or file-like object)
                export_params=True,                         # store the trained parameter weights inside the model file
                opset_version=opset_version,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = input_name_list,              # the model's input names
                output_names = output_name_list,
                dynamic_axes=dynamic_axes
        )

        if self.use_logger_platform:
            file_logger = mlogger.Container()
            file_logger.onnx_file = mlogger.FileLogger('onnx', 'aliyun')
            file_logger.onnx_file.update(os.path.join(self.work_dir, f'{prefix}.onnx'))

        if self.cfg.export.get('deploy', None) is None:
            return

        # 基于目标引擎，转换onnx模型
        # deploy=dict(
        #     engine='rknn',      # rknn,snpe,tensorrt,tnn
        #     device='rk3568',    # rk3568/rk3588,qualcomm,nvidia,mobile
        #     preprocess=dict(
        #         mean_values='0,0,0',        # mean values
        #         std_values='255,255,255'    # std values
        #     ),
        #     quantize=False,                 # is quantize
        #     calibration=dict(...)           # calibration dataset config
        # )
        target_engine = self.cfg.export.deploy.engine  # rknn,snpe,tensorrt,tnn

        if target_engine == 'rknn':
            target_device = self.cfg.export.deploy.device  # rk3568/rk3588
            mean_values = self.cfg.export.deploy.preprocess.mean_values        # 0,0,0
            std_values = self.cfg.export.deploy.preprocess.std_values          # 255,255,255
            if isinstance(mean_values, list) or isinstance(mean_values, tuple):
                mean_values = ','.join(f'{value}' for value in mean_values)
            if isinstance(std_values, list) or isinstance(std_values, tuple):
                std_values = ','.join(f'{value}' for value in std_values)
            print(f'using mean_values {mean_values}, std_values {std_values}')
            if self.cfg.export.deploy.get('quantize', False):
                # int8
                # 生成校准数据
                if not os.path.exists(os.path.join(self.work_dir, 'calibration-images')):
                    dataset = build_dataset(self.cfg.export.deploy.calibration)
                    os.makedirs(os.path.join(self.work_dir, 'calibration-images'))

                    count = 0
                    for sample in dataset:
                        image = sample['image']
                        
                        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                            print('calibration data not correct.')
                            return 

                        cv2.imwrite(os.path.join(self.work_dir, 'calibration-images', f'{count}.png'), image)
                        count += 1

                        if self.cfg.export.deploy.get('calibration_size', -1) > 0:
                            if count > self.cfg.export.deploy.calibration_size:
                                break

                # 开始转模型
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {onnx_file_path} /tmp/onnx/')
                shutil.copytree(os.path.join(self.work_dir, 'calibration-images'), '/tmp/onnx/calibration-images')

                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --image-folder=./calibration-images --quantize --device={target_device} --mean-values={mean_values} --std-values={std_values}')
                os.system(f'cp -r /tmp/onnx/rknn/* {self.work_dir} ; rm -rf /tmp/onnx/')
            else:
                # fp16
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {onnx_file_path} /tmp/onnx/')
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --device={target_device} --mean-values={mean_values} --std-values={std_values}')
                os.system(f'cp -r /tmp/onnx/rknn/* {self.work_dir} ; rm -rf /tmp/onnx/')
        elif target_engine == 'snpe':
            if self.cfg.export.deploy.quantize:
                # 生成校准数据
                dataset = build_dataset(self.cfg.export.deploy.calibration)
                if not os.path.exists(os.path.join(self.work_dir, 'calibration-images')):
                    os.makedirs(os.path.join(self.work_dir, 'calibration-images'))

                count = 0
                for sample in dataset:
                    image = sample['image']
                    
                    # 注意这里需要保证image已经是减过均值，除过方差的
                    if image.dtype != np.float32:
                        print('For snpe int8 deploy. calibration data must be float32 (finish preprocess)')
                        return

                    np.save(os.path.join(self.work_dir, 'calibration-images', f'{count}.npy'), image)
                    count += 1

                # npu
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/snpe ; cp {onnx_file_path} /tmp/onnx/')
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix} --quantize --npu --data-folder=calibration-images')
                os.system(f'cp -r /tmp/onnx/snpe/* {self.work_dir} ; rm -rf /tmp/onnx/')
            else:
                # other 
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/snpe ; cp {onnx_file_path} /tmp/onnx/')                    
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix}')
                os.system(f'cp -r /tmp/onnx/snpe/* {self.work_dir} ; rm -rf /tmp/onnx/')
        elif target_engine == 'tensorrt':
            if self.cfg.export.deploy.quantize:
                # 生成校准数据
                dataset = build_dataset(self.cfg.export.deploy.calibration)
                if not os.path.exists(os.path.join(self.work_dir, 'calibration-images')):
                    os.makedirs(os.path.join(self.work_dir, 'calibration-images'))

                count = 0
                for sample in dataset:
                    image = sample['image']
                    
                    # 注意这里需要保证image已经是减过均值，除过方差的
                    if image.dtype != np.float32:
                        print('For snpe int8 deploy. calibration data must be float32 (finish preprocess)')
                        return

                    np.save(os.path.join(self.work_dir, 'calibration-images', f'{count}.npy'), image)
                    count += 1

                # int8
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/tensorrt ; cp {onnx_file_path} /tmp/onnx/')                    
                os.system(f'cd /tmp/onnx/ ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace --gpus all tensorrtconvert bash convert.sh --i={prefix}.onnx --o=./tensorrt/{prefix} --quantize --data-folder=calibration-images')
                os.system(f'cp -r /tmp/onnx/tensorrt/* {self.work_dir} ; rm -rf /tmp/onnx/')
            else:
                # fp16
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/tensorrt ; cp {onnx_file_path} /tmp/onnx/')                      
                os.system(f'cd /tmp/onnx/ ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace --gpus all tensorrtconvert bash convert.sh --i={prefix}.onnx --o=./tensorrt/{prefix}')
                os.system(f'cp -r /tmp/onnx/tensorrt/* {self.work_dir} ; rm -rf /tmp/onnx/')
        elif target_engine == 'tnn':
            print('Only use tnn for mobile gpu/cpu deploy')
            onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
            os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/tnn ; cp {onnx_file_path} /tmp/onnx/')                 
            os.system(f'cd /tmp/onnx/ ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace tnnconvert bash convert.sh --i={prefix}.onnx --o=./tnn/{prefix}')
            os.system(f'cp -r /tmp/onnx/tnn/* {self.work_dir} ; rm -rf /tmp/onnx/')
