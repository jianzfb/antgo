import os
import sys
import argparse
import numpy as np
import onnx

# python3 convert.py --i xx.onnx --outnode --innode --quantize --gpu --cpu --npu --o name --image-folder --data-folder

def main():
    parser = argparse.ArgumentParser(description=f'SNPE-CONVERT')
    parser.add_argument('--i', type=str, help='onnx model file')
    parser.add_argument('--o', type=str, default='model', help='device model file')
    parser.add_argument('--outnode', type=str, help='out node name list')
    parser.add_argument('--innode', type=str, default='', help='in node name list')
    parser.add_argument('--quantize', action='store_true', help='int8')
    parser.add_argument('--gpu', action='store_true', help='target device GPU')
    parser.add_argument('--cpu', action='store_true', help='target device CPU')
    parser.add_argument('--npu', action='store_true', help='target device NPU')
    parser.add_argument('--qat', action='store_true', help='quantize mode')
    
    parser.add_argument('--image-folder', type=str, default='image folder (for calibration)', help='image file folder')
    parser.add_argument('--data-folder', type=str, default='data folder (for calibration)', help='data file folder (binary float)')
    parser.add_argument('--version', type=str, default="1.0")               # 模型版本
    
    args = parser.parse_args()
    if not os.path.exists(args.i):
        print(f'Dont exist onnx model file {args.i}')
        return -1

    xmodel = onnx.load(args.i)
    xgraph = xmodel.graph
    output_num = len(xgraph.output)
    model_outnode_list = []
    for output_i in range(output_num):
        output_name = xgraph.output[output_i].name
        model_outnode_list.append(output_name)
    
    if args.quantize:
        # 检查校准数据是否存在
        if not os.path.exists(args.data_folder):
            print(f'Dont exist calibration data folder {args.data_folder}')
            return -1
        
        # 将校准数据文件夹，整理成文件索引
        calibration_data_file_list = []
        for file_name in os.listdir(args.data_folder):
            if file_name[0] == '.':
                continue
            
            # 注意这里的，文件是npy，具体数据是float32
            data = np.load(os.path.join(args.data_folder, file_name))
            
            # 获得二进制数据，重新进行保存
            data.tofile(os.path.join(args.data_folder, f'{file_name}.raw'))
            calibration_data_file_list.append(os.path.join(args.data_folder, f'{file_name}.raw'))

        if len(calibration_data_file_list) == 0:
            print(f'Dont have enough calibration data.')
            return -1

        with open('./snpe_calibration_data.txt', 'w') as fp:
            for line_content in calibration_data_file_list:
                fp.write(f'{line_content}\n')

        # 导出量化模型，涉及CPU，NPU
        if args.qat:
            # 在线量化模式，量化信息已经存在onnx模型中
            pass
        else:
            # 离线量化模式，需要使用校准数据进行量化信息计算
            cmd_str = f'snpe-onnx-to-dlc -i {args.i} -o quant.dlc'
            os.system(cmd_str)

            cmd_str = f'snpe-dlc-quantize --input_dlc quant.dlc --output_dlc {args.o}.{args.version}.dlc --input_list ./snpe_calibration_data.txt'
            if args.npu:
                cmd_str += ' --enable_hta'

            os.system(cmd_str)
    else:
        # 导出浮点模型（fp16），涉及GPU,CPU
        cmd_str = f'snpe-onnx-to-dlc -i {args.i} -o {args.o}.{args.version}.dlc'
        for outnode in model_outnode_list:
            cmd_str += f" --out_node={outnode}"
        os.system(cmd_str)

    return 0


if __name__ == '__main__':
    main()
