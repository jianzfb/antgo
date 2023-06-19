from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.dataset import build_dataset
import numpy as np
import cv2
from thop import profile
import json
from collections import OrderedDict
import re

class Exporter(object):
    def __init__(self, cfg, work_dir):
        if isinstance(cfg, dict):
            self.cfg = Config.fromstring(json.dumps(cfg), '.json')
        else:
            self.cfg = cfg
        self.work_dir = work_dir

    def export(self, input_tensor_list, input_name_list, output_name_list=None, checkpoint=None, model_builder=None, prefix='model', opset_version=12, is_convert_to_deploy=False, revise_keys=[], strict=True):
        model = None
        if model_builder is not None:
            model = model_builder()
        else:
            model = build_model(self.cfg.model)

        # 加载checkpoint
        if checkpoint is not None:
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
        model.forward = model.onnx_export
        model = model.to('cpu')
        if isinstance(input_tensor_list, list):
            for i in range(len(input_tensor_list)):
                input_tensor_list[i] = input_tensor_list[i].to('cpu')

        flops, params = profile(model, inputs=input_tensor_list)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # Export the model
        torch.onnx.export(
                model,                                      # model being run
                tuple(input_tensor_list),                   # model input (or a tuple for multiple inputs)
                os.path.join(self.work_dir, f'{prefix}.onnx'),       # where to save the model (can be a file or file-like object)
                export_params=True,                         # store the trained parameter weights inside the model file
                opset_version=opset_version,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = input_name_list,              # the model's input names
                output_names = output_name_list
        )

        # 基于目标引擎，转换onnx模型
        if is_convert_to_deploy:
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
            
            target_engine = self.cfg.deploy.engine  # rknn,snpe,tensorrt,tnn
            target_device = self.cfg.deploy.device  # rk3568/rk3588,qualcomm,nvidia,mobile

            if target_engine == 'rknn':
                mean_values = self.cfg.deploy.preprocess.mean_values        # 0,0,0
                std_values = self.cfg.deploy.preprocess.std_values          # 255,255,255
                print(f'using mean_values {mean_values}, std_values {std_values}')
                if self.cfg.deploy.quantize:
                    # int8
                    # 生成校准数据
                    dataset = build_dataset(self.cfg.deploy.calibration)
                    if not os.path.exists(os.path.join(self.work_dir, 'calibration-images')):
                        os.makedirs(os.path.join(self.work_dir, 'calibration-images'))

                    count = 0
                    for sample in dataset:
                        image = sample['image']
                        
                        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                            print('calibration data not correct.')
                            return 

                        cv2.imwrite(os.path.join(self.work_dir, 'calibration-images', f'{count}.png'), image)
                        count += 1

                    # 开始转模型
                    onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/rknn && cp {onnx_file_path} /tmp/onnx/')
                    os.system(f'cd /tmp/onnx && docker run -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --image-folder=calibration-images --quantize --device={target_device} --mean-values={mean_values} --std-values={std_values}')
                    os.system(f'cp -r /tmp/onnx/rknn/* {self.work_dir} && rm -rf /tmp/onnx/')
                else:
                    # fp16
                    onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/rknn && cp {onnx_file_path} /tmp/onnx/')
                    os.system(f'cd /tmp/onnx && docker run -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --device={target_device} --mean-values={mean_values} --std-values={std_values}')
                    os.system(f'cp -r /tmp/onnx/rknn/* {self.work_dir} && rm -rf /tmp/onnx/')
            elif target_engine == 'snpe':
                if self.cfg.deploy.quantize:
                    # 生成校准数据
                    dataset = build_dataset(self.cfg.deploy.calibration)
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
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/snpe && cp {onnx_file_path} /tmp/onnx/')
                    os.system(f'cd /tmp/onnx && docker run -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix} --quantize --npu --data-folder=calibration-images')
                    os.system(f'cp -r /tmp/onnx/snpe/* {self.work_dir} && rm -rf /tmp/onnx/')
                else:
                    # other 
                    onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/snpe && cp {onnx_file_path} /tmp/onnx/')                    
                    os.system(f'cd /tmp/onnx && docker run -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix}')
                    os.system(f'cp -r /tmp/onnx/snpe/* {self.work_dir} && rm -rf /tmp/onnx/')
            elif target_engine == 'tensorrt':
                if self.cfg.deploy.quantize:
                    # 生成校准数据
                    dataset = build_dataset(self.cfg.deploy.calibration)
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
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/tensorrt && cp {onnx_file_path} /tmp/onnx/')                    
                    os.system(f'cd /tmp/onnx/ && docker run -v $(pwd):/workspace --gpus all tensorrtconvert bash convert.sh --i={prefix}.onnx --o=./tensorrt/{prefix} --quantize --data-folder=calibration-images')
                    os.system(f'cp -r /tmp/onnx/tensorrt/* {self.work_dir} && rm -rf /tmp/onnx/')
                else:
                    # fp16
                    onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                    os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/tensorrt && cp {onnx_file_path} /tmp/onnx/')                      
                    os.system(f'cd /tmp/onnx/ && docker run -v $(pwd):/workspace --gpus all tensorrtconvert bash convert.sh --i={prefix}.onnx --o=./tensorrt/{prefix}')
                    os.system(f'cp -r /tmp/onnx/tensorrt/* {self.work_dir} && rm -rf /tmp/onnx/')
            elif target_engine == 'tnn':
                print('Only use tnn for mobile gpu deploy')
                onnx_file_path = os.path.join(self.work_dir, f'{prefix}.onnx')
                os.system(f'mkdir /tmp/onnx && mkdir /tmp/onnx/tnn && cp {onnx_file_path} /tmp/onnx/')                 
                os.system(f'cd /tmp/onnx/ && docker run -v $(pwd):/workspace tnnconvert bash convert.sh --i={prefix}.onnx --o=./tnn/{prefix}')
                os.system(f'cp -r /tmp/onnx/tnn/* {self.work_dir} && rm -rf /tmp/onnx/')
