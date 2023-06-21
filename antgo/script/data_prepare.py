from fileinput import filename
import sys
import os
import json
import logging
from antgo.utils import args
system_path = os.path.join(os.path.abspath(os.curdir),'system.py')
os.system(f'ln -sf {system_path}  {os.path.dirname(os.path.realpath(__file__))}/system.py')
from antgo.framework.helper.fileio import *
from antgo.framework.helper.utils import Config
from system import *


args.DEFINE_nn_args()

def main():
    nn_args = args.parse_args()
    # 下载扩展配置中的数据到本地
    if os.path.exists(nn_args.extra_config):
        with open(nn_args.extra_config, 'r') as fp:
            extra_config = Config.fromstring(fp.read(),'.json')

        # extra_config 格式为项目信息格式
        # step3.1: 数据相关下载 (默认TFRECORD是antgo默认标准打包格式)
        extra_dataset_train_label = extra_config['source'].get('label', [])
        extra_dataset_train_pseudo_label = extra_config['source'].get('pseudo-label', [])
        extra_dataset_train_unlabel = extra_config['source'].get('unlabel', [])

        # 创建dataset-storage文件夹
        if not os.path.exists('dataset-storage'):
            os.mkdir('dataset-storage')

        # 有标签数据下载，伪标签数据下载，无标签数据下载
        for data_stage, data_info_list in zip(
            ['label', 'pseudo-label', 'unlabel'], 
            [extra_dataset_train_label, extra_dataset_train_pseudo_label, extra_dataset_train_unlabel]):

            # 下载相关数据，到训练集群
            if not os.path.exists(f"dataset-storage/{data_stage}/"):
                os.makedirs(f"dataset-storage/{data_stage}/")

            for data_info in data_info_list:
                if data_info['status'] and data_info['address'] != '':
                    if data_info['address'].startswith('/'):
                        # 本地路径
                        data_folder = os.path.dirname(data_info['address'])
                        target_data_folder = f"dataset-storage/{data_stage}"

                        check_prefix = data_info['address'].split('/')[-1]
                        for file_name in os.listdir(data_folder):
                            if os.path.exists(f'{target_data_folder}/{file_name}'):
                                # 目标文件已经存在，直接跳过
                                continue

                            if file_name.startswith(check_prefix):
                                os.system(f'ln -s {data_folder}/{file_name} {target_data_folder}/{file_name}')                    
                    else:
                        check_prefix = data_info['address'].split('/')[-1]
                        is_existed = False
                        for file_name in os.listdir(f"dataset-storage/{data_stage}"):
                            if file_name.startswith(check_prefix):
                                is_existed = True
                                break

                        if is_existed:
                            continue

                        if not data_info['address'].endswith('*'):
                            data_info['address'] += '*'
                        status = file_client_get(data_info['address'], f"dataset-storage/{data_stage}")
                        if not status:
                            logging.error(f'Download {data_info["address"]} error.')

    # 下载配置中的数据到本地
    if not os.path.exists(nn_args.config):
        # 查找位置1
        config_file_path = os.path.join(nn_args.exp.split('.')[0], 'config.py')
        if not os.path.exists(config_file_path):
            config_file_path = ''

        # 查找位置2
        if config_file_path == '':
            config_file_path = os.path.join(nn_args.exp.split('.')[0], 'configs', 'config.py')
            if not os.path.exists(config_file_path):
                config_file_path = ''
        nn_args.config = config_file_path

    if os.path.exists(nn_args.config):
        # 创建dataset-storage文件夹
        if not os.path.exists('dataset-storage'):
            os.mkdir('dataset-storage')

        cfg = Config.fromfile(nn_args.config)
        for data_stage in ['train', 'val', 'test']:
            if getattr(cfg.data, data_stage, None) is None:
                continue
            if getattr(getattr(cfg.data, data_stage), 'data_path_list', None) is None:
                continue

            if not os.path.exists(f"dataset-storage/{data_stage}/"):
                os.makedirs(f"dataset-storage/{data_stage}/")

            for data_record_path in getattr(cfg.data, data_stage).data_path_list:
                if data_record_path.startswith('/') or data_record_path.startswith('./'):
                    # 本地路径
                    data_folder = os.path.dirname(data_record_path)
                    target_data_folder = f"dataset-storage/{data_stage}"

                    check_prefix = data_record_path.split('/')[-1]
                    for file_name in os.listdir(data_folder):
                        if os.path.exists(f'{target_data_folder}/{file_name}'):
                            # 目标文件已经存在，直接跳过
                            continue

                        if file_name.startswith(check_prefix):
                            os.system(f'ln -s {data_folder}/{file_name} {target_data_folder}/{file_name}')                    
                else:
                    check_prefix = data_record_path.split('/')[-1]
                    is_existed = False
                    for file_name in os.listdir(f"dataset-storage/{data_stage}"):
                        if file_name.startswith(check_prefix):
                            is_existed = True
                            break

                    if is_existed:
                        continue

                    if not data_record_path.endswith('*'):
                        data_record_path += '*'
                    status = file_client_get(data_record_path, f"dataset-storage/{data_stage}")
                    if not status:
                        logging.error(f'Download {data_record_path} error.')

    # 检查dataset-storage是否为空
    if os.listdir('dataset-storage') == 0:
        os.system('rm -rf dataset-storage')

    # 下载checkpoint到本地
    if nn_args.checkpoint is not None and nn_args.checkpoint != '':
        if not os.path.exists('checkpoint-storage'):
            os.mkdir('checkpoint-storage')

        if not nn_args.checkpoint.startswith('/') and not nn_args.checkpoint.startswith('./'):
            status = file_client_get(nn_args.checkpoint, f"checkpoint-storage")
            if not status:
                logging.error(f'Download checkpoint fail.')

    if nn_args.resume_from is not None and nn_args.resume_from != '':
        if not os.path.exists('checkpoint-storage'):
            os.mkdir('checkpoint-storage')

        if not nn_args.checkpoint.startswith('/') and not  nn_args.checkpoint.startswith('./'):
            status = file_client_get(nn_args.resume_from, f"checkpoint-storage")
            if not status:
                logging.error(f'Download checkpoint fail.')

    
if __name__ == "__main__":
    main()