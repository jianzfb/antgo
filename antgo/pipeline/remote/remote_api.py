# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : remote_api.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo import config
import os
import cv2
import base64
import numpy as np


class RemoteApiOp(object):
    def __init__(self, server_name, function_name='', **kwargs):
        token = None
        self.rpc = None
        if os.path.exists('./.token'):
            with open('./.token', 'r') as fp:
                token = fp.readline()

        if token is None or token == '':
            config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
            config.AntConfig.parse_xml(config_xml)
            token = getattr(config.AntConfig, 'server_user_token', '')
        if token == '' or token is None:
            print('No valid vibstring token, directly return')
            return

        self.function_name = function_name
        server_ip = kwargs.get('server_ip', 'research.vibstring.com')
        if 'server_ip' in kwargs:
            kwargs.pop('server_ip')
        server_port = kwargs.get('server_port', 80)
        if 'server_port' in kwargs:
            kwargs.pop('server_port')
        self.rpc = HttpRpc("v1", server_name, server_ip, server_port, token=token)
        # self.rpc = HttpRpc("v1", '', server_ip, server_port, token=token)
        self.rpc.headers.update(
            {
                'Content-Type': 'application/json'
            }
        )

        self._index = None
        self.input_config = None
        self.output_config = None
        self.kwargs = kwargs

    def __call__(self, *args):
        if self.rpc is None:
            return

        input_names = self._index[0]
        if isinstance(input_names, str):
            input_names = [input_names]
        output_names = self._index[1]
        if isinstance(output_names, str):
            output_names = [output_names]

        if self.input_config is None:
            self.input_config = {}
            for input_name in input_names:
                self.input_config[input_name] = self.kwargs[input_name]
                self.kwargs.pop(input_name)
        if self.output_config is None:
            self.output_config = {}
            for output_name in output_names:
                self.output_config[output_name] = self.kwargs[output_name]
                self.kwargs.pop(output_name)

        input_req = {}
        for index, value in enumerate(args):
            arg_name = input_names[index]
            arg_type = self.input_config[arg_name]
            if arg_type == 'image':
                # 转换成base64编码
                _, buffer = cv2.imencode('.png', value)
                input_req[arg_name] = base64.b64encode(buffer.tobytes()).decode()
            elif arg_type in ['video', 'file']:
                # 文件类型
                assert(value, str)
                if not os.path.exists(value):
                    continue
                # 上传文件，获得服务端地址
                file_id = self.rpc.file.upload(file=value)[0]
                input_req[arg_name] = file_id
            else:
                # 其他类型均按照字符串对待
                input_req[arg_name] = value

        # 补充参数
        input_req.update(self.kwargs)

        content = getattr(self.rpc, self.function_name).execute.post(**input_req)
        out_values = []
        for arg_name, arg_value in content.items():
            if arg_name not in output_names:
                continue

            arg_type = self.output_config[arg_name]
            if arg_type == 'image':
                # 图像类型
                content = self.rpc.file.download(file_folder='./temp', file_name=arg_value, is_bytes_io=True)
                if content['status'] == 'ERROR':
                    out_values.append(None)
                    continue
                image = cv2.imdecode(np.frombuffer(content['file'], dtype='uint8'), cv2.IMREAD_UNCHANGED)
                out_values.append(image)
            elif arg_type in ['video', 'file']:
                # 文件类型
                content = self.rpc.file.download(file_folder='./temp', file_name=arg_value)
                out_values.append(os.path.join('./temp', content['file']))
            else:
                # 其他类型
                out_values.append(arg_value)
        if len(out_values) == 0:
            return None

        return out_values[0] if len(out_values) == 1 else out_values
