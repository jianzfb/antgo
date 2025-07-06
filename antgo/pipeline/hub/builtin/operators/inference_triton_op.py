# -*- coding: UTF-8 -*-
# @Time    : 2025/7/5 22:42
# @File    : triton_model_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
import logging
import os
import numpy as np
import cv2
import base64


@register
class inference_triton_op(object):
    def __init__(self, url, model_name, version=1, config=None):
        self.url = f'{url}/v2/models/{model_name}/versions/{version}/infer'
        # {
        #     "inputs": [
        #         {
        #             "name": xxx,
        #             "shape": xxx,
        #             "datatype": "BYTES",
        #             "data": []
        #         }
        #     ],
        #     "outputs": [
        #         {"name": "ENSEMBLE_OUTPUT_BOXES"}, 
        #         {"name": "ENSEMBLE_OUTPUT_SCORES"},
        #         {"name": "ENSEMBLE_OUTPUT_CLASSES"}
        #     ]
        # }
        self.config = {} if config is None else config
        self.typemap = {
            'INT32': np.int32,
            'FP32': np.float32,
        }

    def __call__(self, *args):
        # TODO，将来支持任意数据类型
        # 目前仅支持图像输入
        data_config = copy.deepcopy(self.config)
        if 'outputs' not in data_config:
            output_index = self._index[1]
            if isinstance(output_index, str):
                output_index = [output_index]
            
            data_config['outputs'] = []
            for output_name in output_index:
                data_config['outputs'].append({
                    'name': output_name
                })
        if 'inputs' not in data_config:
            input_index = self._index[0]
            if isinstance(input_index, str):
                input_index = [input_index]
            
            data_config['inputs'] = []
            for input_name in input_index:
                data_config['inputs'].append({
                    'name': input_name
                })

        for input_config, input_data in zip(data_config["inputs"], args):
            if input_data.dtype == np.uint8 and ((len(input_data.shape) == 2) or (len(input_data.shape) == 3 and (input_data.shape[-1] == 3 or input_data.shape[-1] == 4))):
                # 图像数据（灰度图、3通道图、4通道图）
                input_config['datatype'] = "BYTES"
                success, encoded_image = cv2.imencode('.webp', input_data)
                base64_image = base64.b64encode(encoded_image).decode('utf-8')
                input_config['data'] = [base64_image]
                input_config['shape'] = [1]
            elif input_data.dtype == np.uint8 and len(input_data.shape) == 4:
                # 多图像数据
                input_config['datatype'] = "BYTES"
                input_config['data'] = []
                input_config['shape'] = [input_data.shape[0]]
                for image_data in input_data:
                    success, encoded_image = cv2.imencode('.webp', input_data)
                    base64_image = base64.b64encode(encoded_image).decode('utf-8')
                    input_config['data'].append(base64_image)
            else:
                # 非图像数据
                print('Support in them future')
                pass

        response = requests.post(self.url, json=data_config)
        if response.status_code != 200:
            print(f"Request failed: {response.status_code}")
            return [None for _ in range(len(data_config['outputs']))]
        result = response.json()
        if 'outputs' not in result or not result['outputs']:
            print("No outputs in response")
            return [None for _ in range(len(data_config['outputs']))]

        outputs = {out['name']: out for out in result['outputs']}
        output_list = []
        for output_config in data_config['outputs']:
            output_name = output_config['name']
            output_list.append(np.array(outputs[output_name]['data'], dtype=self.typemap[outputs[output_name]['datatype']]).reshape(outputs[output_name]['shape']))

        return tuple(output_list) if len(output_list) > 1 else output_list[0]
