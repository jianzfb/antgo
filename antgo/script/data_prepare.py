import sys
import os
import json
import logging
from antgo.utils import args
system_path = os.path.join(os.path.abspath(os.curdir),'system.py')
os.system(f'ln -sf {system_path}  {os.path.dirname(os.path.realpath(__file__))}/system.py')
from antgo.ant import environment
from antgo.framework.helper.utils import Config
from system import *


args.DEFINE_nn_args()

def main():
    nn_args = args.parse_args()
    if os.path.exists(nn_args.extra_config):
        with open(nn_args.extra_config, 'r') as fp:
            extra_config = Config.fromstring(fp.read(),'.json')

        # extra_config 格式为项目信息格式
        # step3.1: 数据相关下载 (默认TFRECORD是antgo默认标准打包格式)
        extra_dataset_train_label = extra_config['source']['label']
        extra_dataset_train_pseudo_label = extra_config['source']['pseudo-label']
        extra_dataset_train_unlabel = extra_config['source']['unlabel']

        if not os.path.exists('-dataset-'):
            os.mkdir('-dataset-')

        # 有标签数据下载
        if len(extra_dataset_train_label) > 0:
            # 下载相关数据，到训练集群
            if not os.path.exists(f"-dataset-/label/"):
                os.makedirs(f"-dataset-/label/")

            for data_info in extra_dataset_train_label:
                if data_info['status'] and data_info['address'] != '':
                    local_path = f"-dataset-/label/{data_info['address'].split('/')[-1]}"
                    status = environment.hdfs_client.get(data_info['address'], local_path)                        
                    if not status:
                        logging.error(f'Download {data_info["address"]} error.')
                        continue

        # 伪标签数据下载
        if len(extra_dataset_train_pseudo_label) > 0:
            # 下载相关数据，到训练集群
            if not os.path.exists(f"-dataset-/pseudo-label/"):
                os.makedirs(f"-dataset-/pseudo-label/")

            for data_info in extra_dataset_train_pseudo_label:          
                if data_info['status'] and data_info['address'] != '':
                    local_path = f"-dataset-/pseudo-label/{data_info['address'].split('/')[-1]}"
                    status = environment.hdfs_client.get(data_info['address'], local_path)
                    if not status:
                        logging.error(f'Download {data_info["address"]} error.')
                        continue       

        # 无标签数据下载
        if len(extra_dataset_train_unlabel)> 0:
            # 下载相关数据，到训练集群
            if not os.path.exists(f"-dataset-/unlabel/"):
                os.makedirs(f"-dataset-/unlabel/")

            for data_info in extra_dataset_train_unlabel:
                if data_info['status'] and data_info['address'] != '':
                    local_path = f"-dataset-/unlabel/{data_info['address'].split('/')[-1]}"
                    status = environment.hdfs_client.get(data_info['address'], local_path)
                    if not status:
                        logging.error(f'Download {data_info["address"]} error.')
                        continue       

    if os.path.exists(nn_args.config):
        if not os.path.exists('-dataset-'):
            os.mkdir('-dataset-')
        
        cfg = Config.fromfile(nn_args.config)
        if getattr(cfg.data, 'train', None):
            if not os.path.exists(f"-dataset-/train/"):
                os.makedirs(f"-dataset-/train/")
            
            for data_record_path in cfg.data.train.data_path_list:
                local_path = f"-dataset-/train/{data_record_path.split('/')[-1]}"
                if data_record_path.startswith('hdfs'):
                    status = environment.hdfs_client.get(data_record_path, local_path)

                    if not status:
                        logging.error(f'Download {data_record_path} error.')
                
        if getattr(cfg.data, 'val', None):
            if not os.path.exists(f"-dataset-/val/"):
                os.makedirs(f"-dataset-/val/")
            
            for data_record_path in cfg.data.val.data_path_list:
                local_path = f"-dataset-/val/{data_record_path.split('/')[-1]}"
                if data_record_path.startswith('hdfs'):
                    status = environment.hdfs_client.get(data_record_path, local_path)

                    if not status:
                        logging.error(f'Download {data_record_path} error.')
        
        if getattr(cfg.data, 'test', None):
            if not os.path.exists(f"-dataset-/test/"):
                os.makedirs(f"-dataset-/test/")

            for data_record_path in cfg.data.test.data_path_list:
                local_path = f"-dataset-/test/{data_record_path.split('/')[-1]}"
                if data_record_path.startswith('hdfs'):
                    status = environment.hdfs_client.get(data_record_path, local_path)

                    if not status:
                        logging.error(f'Download {data_record_path} error.')
        
        
            
    
if __name__ == "__main__":
    main()