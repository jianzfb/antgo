import argparse
import os.path as osp
import warnings
from functools import partial
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.runner.checkpoint import load_checkpoint
from antgo.framework.helper.tools.util import *
from antgo.framework.helper.models.builder import *
from antgo.framework.helper.utils import Config
import numpy as np
import onnx
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ANTGO models to ONNX')
    parser.add_argument('--config', type=str, default='/root/paddlejob/workspace/env_run/portrait/InterHand26M/main/test_config.py', help='test config file path')
    parser.add_argument('--ext-module', type=str, default='ext_module.py', help='introduce ext module py file')
    parser.add_argument('--checkpoint', type=str, default='/root/paddlejob/workspace/env_run/portrait/InterHand26M/main/main/latest.pth', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 224, 224],
        help='input image size')
    parser.add_argument(
        '--input-name',
        type=str,
        default='input'
    )
    parser.add_argument(
        '--output-names',
        type=str,
        default='output'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.ext_module != '':
        # 导入ext module
        print(f'load ext model from {args.ext_module}')
        load_extmodule(args.ext_module)

    print('build model')
    cfg.model.train_cfg = None
    model = \
        build_model(cfg.model, test_cfg=cfg.get('test_cfg', None))
    print(f"load chekcpoint from {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    input_name = args.input_name
    output_names = args.output_names.split(',')
    input_data = np.ones(args.shape, dtype=np.float32)
    
    print('export onnx')
    torch.onnx.export(
        model,
        input_data,
        args.output_file,
        input_names=[input_name],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=True,
        opset_version=args.opset_version)
    

