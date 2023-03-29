# (1) 训练过程
# 多机多卡训练
# TODO
# 单机多卡训练(4卡运行)
# bash launch.sh ../cifar10/main.py 4 --exp=xxx --no-validate --process=train
# 单机1卡训练(可以自定义使用第几块卡 --gpu-id=0)
# python3 ./cifar10/main.py --exp=xxx --gpu-id=0 --no-validate --process=train
# 单机CPU训练(仅用于调试)
# python3 ./cifar10/main.py --exp=xxx --gpu-id=-1 --no-validate --process=train

# (2) 评估过程
# 单机多卡评估(4卡运行)
# bash launch.sh ./cifar10/main.py 4 --exp=xxx --checkpoint=yyy --process=test
# 单机1卡评估(可以自定义使用第几块卡 --gpu-id=0)
# python3 ./cifar10/main.py --exp=xxx --checkpoint=yyy --gpu-id=0 --process=test
# 单机CPU测试(仅用于调试)
# python3 ./cifar10/main.py --exp=xxx --checkpoint=yyy --gpu-id=-1 --process=test

# (3) 模型导出过程
# python3 ./cifar10/main.py --exp=xxx --checkpoint=yyy --process=export

# 1.step 通用模块
import shutil
import sys
import os
import json
import logging

system_path = os.path.join(os.path.abspath(os.curdir),'system.py')
os.system(f'ln -sf {system_path}  {os.path.dirname(os.path.realpath(__file__))}/system.py')
import torch
from antgo.utils import args
from antgo.framework.helper.trainer import *
from antgo.framework.helper.tester import *
from antgo.framework.helper.activelearning import Activelearning
from antgo.framework.helper.exporter import *
from antgo.framework.helper.models.detectors import *
from antgo.framework.helper.dataset.pipelines import *
from antgo.framework.helper.utils import Config
from antgo.ant import environment
import json

# 2.step 导入自定义系统后台(包括hdfs后端,KV后端)
from system import *

# 3.step 导入扩展模块 (包括自定义的模型,数据集,,等)
from models import *
from metrics import *
from dataset import *
from hooks import *

# 4.step 定义shell参数
# 4.1.step 自定义扩展参数
# 例子：
# args.DEFINE_string('root_hdfs', '', 'label dataset hdfs')

# 4.2.step 定义标准nn参数
args.DEFINE_nn_args()

def main():
    # step1: 加载参数
    nn_args = args.parse_args()
    args.print_args(nn_args)
    assert(nn_args.exp != '')
    # step2: 加载配置文件
    if not os.path.exists(nn_args.config):
        here_dir = os.path.dirname(os.path.realpath(__file__))
        # step2.1: 查找位置1
        config_file_path = os.path.join(here_dir, 'config.py')
        if not os.path.exists(config_file_path):
            config_file_path = ''

        # step2.2: 查找位置2
        if config_file_path == '':
            config_file_path = os.path.join(here_dir, 'configs', 'config.py')
            if not os.path.exists(config_file_path):
                config_file_path = ''

        nn_args.config = config_file_path
    if nn_args.config == '':
        print('Couldnt find correct config file.')
        return

    cfg = Config.fromfile(os.path.join(nn_args.exp, nn_args.config))
    if 'checkpoint_config' in cfg:
        cfg.checkpoint_config['out_dir'] = os.path.join(cfg.checkpoint_config['out_dir'], nn_args.exp)
    if 'evaluation' in cfg:
        cfg.evaluation['out_dir'] = os.path.join(cfg.evaluation['out_dir'], nn_args.exp)

    # step2.1 检查补充配置
    if nn_args.extra_config is not None and nn_args.extra_config != '':
        if os.path.exists(nn_args.extra_config):
            with open(nn_args.extra_config, 'r') as fp:
                extra_config = Config.fromstring(fp.read(),'.json')
            
            # extra_config 格式为项目信息格式
            # step2.1.1: 数据相关 (默认TFRECORD是antgo默认标准打包格式)
            extra_dataset_train_label = extra_config['source']['label']
            extra_dataset_train_pseudo_label = extra_config['source']['pseudo-label']
            extra_dataset_train_unlabel = extra_config['source']['unlabel']

            if not os.path.exists('-dataset-'):
                os.mkdir('-dataset-')

            label_local_path_list = []
            if len(extra_dataset_train_label) > 0:
                # 下载相关数据，到训练集群
                for data_info in extra_dataset_train_label:
                    if data_info['status']:
                        local_path = f"-dataset-/{data_info['address'].split('/')[-1]}"
                        status = environment.hdfs_client.get(data_info['address'], local_path)                        
                        if not status:
                            logging.error(f'Download {data_info["address"]} error.')
                            continue
                        
                        label_local_path_list.append(local_path)

            pseudo_local_path_list = []
            if len(extra_dataset_train_pseudo_label) > 0:
                # 下载相关数据，到训练集群
                for data_info in extra_dataset_train_pseudo_label:
                    if data_info['status']:
                        local_path = f"-dataset-/{data_info['address'].split('/')[-1]}"
                        status = environment.hdfs_client.get(data_info['address'], local_path)
                        if not status:
                            logging.error(f'Download {data_info["address"]} error.')
                            continue       
                        pseudo_local_path_list.append(local_path)

            unlabel_local_path_list = []
            if len(extra_dataset_train_unlabel)> 0:
                # 下载相关数据，到训练集群
                for data_info in extra_dataset_train_unlabel:
                    if data_info['status']:
                        local_path = f"-dataset-/{data_info['address'].split('/')[-1]}"
                        status = environment.hdfs_client.get(data_info['address'], local_path)
                        if not status:
                            logging.error(f'Download {data_info["address"]} error.')
                            continue       
                        unlabel_local_path_list.append(local_path)
    
            # step2.1.2: 训练方法相关 (包括数据的使用方式)
            if 'data' in extra_config:
                # 扩展数据的使用方式
                cfg.data.train = extra_config['data']['train']
                if 'train_dataloader' in extra_config['data']:
                    cfg.data.train_dataloader = extra_config['data']['train_dataloader']
            
            if not isinstance(cfg.data.train, list):
                # 默认仅支持TFDataset
                # 默认常规训练方式
                cfg.data.train.data_path_list.extend(label_local_path_list)
                cfg.data.train.data_path_list.extend(pseudo_local_path_list)
            else:
                # 默认复杂训练方式（如，半监督训练，蒸馏训练，跨域训练模式）
                # 仅支持list size=2 （默认，0: 标签数据+伪标签数据；1: 无标签数据）
                cfg.data.train[0].data_path_list.extend(label_local_path_list)
                cfg.data.train[0].data_path_list.extend(pseudo_local_path_list)
                
                cfg.data.train[1].data_path_list.extend(unlabel_local_path_list)
            
            if 'model' in extra_config:
                # 更新默认配置
                default_model_cfg = cfg.model
                cfg.model = dict(
                    type=extra_config['model']['type'],
                    model=default_model_cfg,
                    train_cfg=extra_config['model']['train_cfg'] if 'train_cfg' in extra_config['model'] else None,
                    test_cfg=extra_config['model']['test_cfg'] if 'test_cfg' in extra_config['model'] else None,
                    init_cfg=extra_config['model']['init_cfg'] if 'init_cfg' in extra_config['model'] else None
                )

            if 'custom_hooks' in extra_config:
                hooks = extra_config['custom_hooks']
                if not isinstance(hooks, list):
                    hooks = [hooks]
                
                # 更新默认配置
                custom_hooks = getattr(cfg, 'custom_hooks', None)
                if custom_hooks is None:
                    custom_hooks = hooks
                else:
                    if isinstance(custom_hooks, list):
                        custom_hooks.extend(hooks)
                    else:
                        custom_hooks = hooks
            
            if 'optimizer' in extra_config:
                cfg.optimizer = extra_config['optimizer']
            if 'optimizer_config' in extra_config:
                cfg.optimizer_config = extra_config['optimizer_config']
            if 'lr_config' in extra_config:
                cfg.lr_config = extra_config['lr_config']
            
    # step3: 执行指令(训练、测试、模型导出)
    if nn_args.process == 'train':
        # 创建训练过程
        trainer = Trainer(
            cfg, 
            './', 
            nn_args.gpu_id, # 对于多卡运行环境,会自动忽略此参数数值
            distributed=nn_args.distributed, 
            diff_seed=nn_args.diff_seed, 
            deterministic=nn_args.deterministic, 
            find_unused_parameters=nn_args.find_unused_parameters)

        trainer.config_dataloader(with_validate=not nn_args.no_validate)
        trainer.config_model(resume_from=nn_args.resume_from, load_from=nn_args.checkpoint)
        if nn_args.max_epochs < 0:
            # 优先使用外部指定max_epochs
            nn_args.max_epochs = cfg.max_epochs
        print(f'max epochs {nn_args.max_epochs}')

        trainer.start_train(max_epochs=nn_args.max_epochs)
    elif nn_args.process == 'test':
        # 创建测试过程
        print(f'nn_args.distributed {nn_args.distributed}')
        print(f'nn_args.gpu-id {nn_args.gpu_id}')
        print(f'nn_args.checkpoint {nn_args.checkpoint}')
        tester = Tester(
            cfg, 
            './',
            nn_args.gpu_id, # 对于多卡运行环境,会自动忽略此参数数值
            distributed=nn_args.distributed)
        tester.config_model(checkpoint=nn_args.checkpoint)
        tester.evaluate()
    elif nn_args.process == 'activelearning':
        # 创建主动学习过程,挑选等待标注样本
        print(f'nn_args.distributed {nn_args.distributed}')
        print(f'nn_args.gpu-id {nn_args.gpu_id}')
        print(f'nn_args.checkpoint {nn_args.checkpoint}')        
        ac = Activelearning(cfg, './', nn_args.gpu_id, distributed=nn_args.distributed)
        ac.config_model(checkpoint=nn_args.checkpoint, revise_keys=[('^','model.')])
        ac.select()        
    elif nn_args.process == 'export':
        # 创建导出模型过程
        tester = Exporter(cfg, './')
        checkpoint_file_name = nn_args.checkpoint.split('/')[-1].split('.')[0]
        tester.export(
            input_tensor_list=[torch.zeros(shape, dtype=torch.float32) for shape in cfg.export.input_shape_list], 
            input_name_list=cfg.export.input_name_list, 
            output_name_list=cfg.export.output_name_list, 
            checkpoint=nn_args.checkpoint, 
            prefix=f'{nn_args.exp}-{checkpoint_file_name}-model')

if __name__ == "__main__":
    main()