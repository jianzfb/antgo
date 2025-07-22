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

        # 是否使用实验管理
        self.use_exp_manage = False
        if kwargs.get('no_manage', False):
            return
        self.use_exp_manage = True

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

        if checkpoint is None or checkpoint == '':
            logger.error('Missing checkpoint file')
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

        if self.work_dir == './':
            self.work_dir = os.path.dirname(checkpoint)

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # Export the model
        dynamic_axes = None
        if is_dynamic:
            dynamic_axes = {}
            for input_node_name in input_name_list:
                dynamic_axes[input_node_name] = {0: "-1"}
            for output_node_name in output_name_list:
                dynamic_axes[output_node_name] = {0: "-1"}

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

        if self.use_exp_manage:
            # 激活试验管理
            pass

        if self.use_exp_manage and mlogger.is_ready():
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
