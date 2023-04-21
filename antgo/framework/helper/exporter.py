from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from antgo.framework.helper.utils.config import Config
from antgo.framework.helper.runner.builder import *
from antgo.framework.helper.models.builder import *
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

    def export(self, input_tensor_list, input_name_list, output_name_list=None, checkpoint=None, model_builder=None, prefix='model', revise_keys=[]):
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
            model.load_state_dict(state_dict, strict=True)

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
                opset_version=11,                           # the ONNX version to export the model to
                do_constant_folding=True,                   # whether to execute constant folding for optimization
                input_names = input_name_list,              # the model's input names
                output_names = output_name_list
        )
