import os
import sys
import argparse
import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

# python3 convert.py --i xx.onnx --outnode --innode --quantize --gpu --cpu --npu --o name --image-folder --data-folder

def main():
    parser = argparse.ArgumentParser(description=f'SNPE-CONVERT')
    parser.add_argument('--i', type=str, help='onnx model file')
    parser.add_argument('--o', type=str, default='model', help='device model file')
    parser.add_argument('--quantize', action='store_true', help='int8')
    parser.add_argument('--device', choices=['rk3568', 'rk3588'], default='rk3588',help='device type')
    parser.add_argument('--image-folder', type=str, default='image folder (for calibration)', help='image file folder')
    parser.add_argument('--mean-values', type=str, default='0,0,0')         # 仅支持3通道
    parser.add_argument('--std-values', type=str, default='255,255,255')    # 仅支持3通道
    parser.add_argument('--version', type=str, default="1.0")               # 模型版本
        
    args = parser.parse_args()
    if not os.path.exists(args.i):
        print(f'Dont exist onnx model file {args.i}')
        return -1

    # mean values
    mean_values = args.mean_values.split(',')
    mean_values = [float(v) for v in mean_values]

    # auto group
    input_num = len(mean_values) // 3
    mean_values = np.array(mean_values).reshape((input_num, -1)).astype(np.float32).tolist()

    # std values
    std_values = args.std_values.split(',')
    std_values = [float(v) for v in std_values]

    # auto group
    input_num = len(std_values) // 3
    std_values = np.array(std_values).reshape((input_num, -1)).astype(np.float32).tolist()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=mean_values, std_values=std_values, target_platform=args.device)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=args.i)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    if args.quantize:
        # 检查校准数据是否存在
        if not os.path.exists(args.image_folder):
            print(f'Dont exist calibration data folder {args.image_folder}')
            return -1

        # 将校准数据文件夹，整理成文件索引
        calibration_data_file_list = []
        for file_name in os.listdir(args.image_folder):
            if file_name[0] == '.':
                continue

            calibration_data_file_list.append(os.path.join(args.image_folder, file_name))
        if len(calibration_data_file_list) == 0:
            print(f'Dont have enough calibration data.')
            return -1

        with open('./rknn_calibration_data.txt', 'w') as fp:
            for line_content in calibration_data_file_list:
                fp.write(f'{line_content}\n')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=args.quantize, dataset='./rknn_calibration_data.txt')
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')
    else:
        # 导出浮点模型（fp16）
        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=args.quantize, dataset='./rknn_calibration_data.txt')
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

    # Export RKNN model
    print('--> Export rknn model')
    rknn_model_name = f'{args.o}.{args.version}'
    if not rknn_model_name.endswith('.rknn'):
        rknn_model_name = f'{rknn_model_name}.rknn'

    ret = rknn.export_rknn(rknn_model_name)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    return 0


if __name__ == '__main__':
    main()
