import numpy as np
import sys
import argparse
import os
import shutil
import onnx

def main():
    parser = argparse.ArgumentParser(description=f'TNN-CONVERT')
    parser.add_argument('--i', type=str, help='onnx model file')
    parser.add_argument('--o', type=str, default='model', help='device model file')
    parser.add_argument('--innode', type=str, default='', help='in node name list')
    parser.add_argument('--quantize', action='store_true', help='int8')
    parser.add_argument('--data-folder', type=str, default='image folder (for calibration)', help='image file folder')
    parser.add_argument('--version', type=str, default="1.0")               # 模型版本

    args = parser.parse_args()
    if not os.path.exists(os.path.join('/workspace', args.i)):
        print(f'Dont exist onnx model file {args.i}')
        return -1
    
    xmodel = onnx.load(os.path.join('/workspace', args.i))
    xgraph = xmodel.graph
    input_num = len(xgraph.input)
    
    in_args = ''
    for input_i in range(input_num):
        input_name = xgraph.input[input_i].name
        input_shape = ''
        dim_num = len(xgraph.input[input_i].type.tensor_type.shape.dim)
        for dim_i in range(dim_num):
            if dim_i < dim_num - 1:
                input_shape += f'{xgraph.input[input_i].type.tensor_type.shape.dim[dim_i].dim_value},'
            else:
                input_shape += f'{xgraph.input[input_i].type.tensor_type.shape.dim[dim_i].dim_value}'
        
        if input_i < input_num - 1:        
            in_args += f'{input_name}:{input_shape} '
        else:
            in_args += f'{input_name}:{input_shape}' 

    if args.quantize:
        print('TODO support.')
        return -1
    else:
        os.system(f'python3 converter.py onnx2tnn /workspace/{args.i} -optimize -align -o /workspace/')
        model_file_name = args.i.split('/')[-1].replace('.onnx', '.opt')
        tnn_model_name = f'{args.o}.{args.version}'

        os.system(f'mv /workspace/{model_file_name}.tnnmodel /workspace/{tnn_model_name}.tnnmodel')
        os.system(f'mv /workspace/{model_file_name}.tnnproto /workspace/{tnn_model_name}.tnnproto')
        return 0


if __name__ == '__main__':
    main()
