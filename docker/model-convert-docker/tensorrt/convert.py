import numpy as np
import tensorrt as trt
import sys
import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description=f'TENSORRT-CONVERT')
    parser.add_argument('--i', type=str, help='onnx model file')
    parser.add_argument('--o', type=str, default='model', help='device model file')
    parser.add_argument('--quantize', action='store_true', help='int8')
    parser.add_argument('--data-folder', type=str, default='image folder (for calibration)', help='image file folder')
    parser.add_argument('--version', type=str, default="1.0")               # 模型版本

    args = parser.parse_args()
    if not os.path.exists(args.i):
        print(f'Dont exist onnx model file {args.i}')
        return -1
    
    trt_model_name = f'{args.o}.{args.version}'
    if not trt_model_name.endswith('.engine'):
        trt_model_name = f'{trt_model_name}.engine'
    
    if args.quantize:
        if not os.path.exists(args.data_folder):
            print(f'Dont exist calibration data')
            return -1

        shutil.move(args.data_folder, "./calibration_data")
        os.system(f'polygraphy convert {args.i} --int8 --data-loader-script /tools/data_loader.py -o {trt_model_name}')
    else:
        os.system(f'polygraphy convert {args.i} -o {trt_model_name}')


if __name__ == '__main__':
    main()
