# (1) 训练过程
# 多机多卡训练
# TODO
# 单机多卡训练(4卡运行)
# bash launch.sh ./cifar10/main.py 4 --exp=xxx --no-validate --process=train
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
from antgo.framework.helper.dataset import *
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

    cfg = Config.fromfile(nn_args.config)
    if 'checkpoint_config' in cfg:
        cfg.checkpoint_config['out_dir'] = os.path.join(cfg.checkpoint_config['out_dir'], nn_args.exp)
    if 'evaluation' in cfg:
        cfg.evaluation['out_dir'] = os.path.join(cfg.evaluation['out_dir'], nn_args.exp)

    for data_stage in ['train', 'val', 'test']:
        # 实验默认的训练，验证，测试数据
        # 如果配置的是远程数据路径，则在本地寻找对应路径数据
        if not getattr(cfg.data, data_stage, None):
            continue

    for data_stage in ['train', 'val', 'test']:
        # 实验默认的训练，验证，测试数据
        # 如果配置的是远程数据路径，则在本地寻找对应路径数据
        if not getattr(cfg.data, data_stage, None):
            continue

        if isinstance(getattr(cfg.data, data_stage), list):
            data_part_0 = getattr(cfg.data, data_stage)[0]
            data_part_1 = getattr(cfg.data, data_stage)[1]

            if not getattr(data_part_0, 'data_path_list', None):
                continue
            for i in range(len(data_part_0.data_path_list)):
                data_record_path = data_part_0.data_path_list[i]
                if data_record_path.startswith('hdfs') or data_record_path.startswith('http'):
                    try_local_path = f"dataset-storage/{data_stage}/{data_record_path.split('/')[-1]}"
                    data_part_0.data_path_list[i] = try_local_path

            if not getattr(data_part_1, 'data_path_list', None):
                continue
            for i in range(len(data_part_1.data_path_list)):                
                data_record_path = data_part_1.data_path_list[i]
                if data_record_path.startswith('hdfs') or data_record_path.startswith('http'):
                    try_local_path = f"dataset-storage/{data_stage}/{data_record_path.split('/')[-1]}"
                    data_part_1.data_path_list[i] = try_local_path
        else:
            if not getattr(getattr(cfg.data, data_stage), 'data_path_list', None):
                continue
            for i in range(len(getattr(cfg.data, data_stage).data_path_list)):
                data_record_path = getattr(cfg.data, data_stage).data_path_list[i]
                if data_record_path.startswith('hdfs') or data_record_path.startswith('http'):
                    try_local_path = f"dataset-storage/{data_stage}/{data_record_path.split('/')[-1]}"
                    getattr(cfg.data, data_stage).data_path_list[i] = try_local_path   

    # step3 检查补充配置
    if nn_args.extra_config is not None and nn_args.extra_config != '':
        if os.path.exists(nn_args.extra_config):
            with open(nn_args.extra_config, 'r') as fp:
                extra_config = Config.fromstring(fp.read(),'.json')

            # extra_config 格式为项目信息格式
            # step3.1: 数据加载 (默认TFRECORD是antgo默认标准打包格式)
            # 有标签数据,伪标签数据,无标签数据,跨域数据 文件列表
            label_local_path_list = []
            pseudo_local_path_list = []
            unlabel_local_path_list = []
            cross_domain_local_path_list = []
            if os.path.exists('dataset-storage'):
                for sub_folder_name in os.listdir('dataset-storage'):
                    if not os.path.isdir(os.path.join('dataset-storage', sub_folder_name)):
                        continue

                    if sub_folder_name == 'label':
                        file_list = []
                        for data_record_file in os.listdir(os.path.join('dataset-storage', sub_folder_name)):
                            # 同一个数据文件，存在*-index和*-tfrecord
                            finding_file = '-'.join(data_record_file.split('-')[:-1])
                            if finding_file not in file_list:
                                file_list.append(finding_file)
                        
                        for data_record_file in file_list:
                            label_local_path_list.append(os.path.join('dataset-storage', sub_folder_name, data_record_file))
                    elif sub_folder_name == 'pseudo-label':
                        file_list = []
                        for data_record_file in os.listdir(os.path.join('dataset-storage', sub_folder_name)):
                            # 同一个数据文件，存在*-index和*-tfrecord
                            finding_file = '-'.join(data_record_file.split('-')[:-1])
                            if finding_file not in file_list:
                                file_list.append(finding_file)

                        for data_record_file in file_list:
                            pseudo_local_path_list.append(os.path.join('dataset-storage', sub_folder_name, data_record_file))
                    elif sub_folder_name == 'unlabel':
                        file_list = []
                        for data_record_file in os.listdir(os.path.join('dataset-storage', sub_folder_name)):
                            # 同一个数据文件，存在*-index和*-tfrecord
                            finding_file = '-'.join(data_record_file.split('-')[:-1])
                            if finding_file not in file_list:
                                file_list.append(finding_file)

                        for data_record_file in file_list:
                            unlabel_local_path_list.append(os.path.join('dataset-storage', sub_folder_name, data_record_file))

            # step3.2: 数据使用方式扩展
            existed_train_datalist = []
            if 'data' in extra_config:
                # 扩展数据的使用方式 (训练集)
                # 训练数据的使用方式扩展
                if 'train' in extra_config['data']:
                    existed_train_datalist = cfg.data.train.data_path_list
                    cfg.data.train = extra_config['data']['train']
                if 'train_dataloader' in extra_config['data']:
                    cfg.data.train_dataloader = extra_config['data']['train_dataloader']

                # 测试数据的使用方式扩展
                if 'test' in extra_config['data']:
                    cfg.data.test = extra_config['data']['test']
                    cfg.data.test.data_path_list = unlabel_local_path_list
                if 'test_dataloader' in extra_config['data']:
                    cfg.data.test_dataloader = extra_config['data']['test_dataloader']

            # step3.3: 数据文件数扩展
            # 训练集相关
            # 功能性数据类 RepeatDataset, DatasetSamplingByClass
            control_dataset_type_names = ['RepeatDataset', 'DatasetSamplingByClass']
            if not isinstance(cfg.data.train, list):
                # 默认仅支持TFDataset
                # 默认常规训练方式
                concrete_dataset = cfg.data.train
                while True:
                    if concrete_dataset.type in control_dataset_type_names:
                        concrete_dataset = concrete_dataset.dataset
                        continue
                    concrete_dataset.data_path_list.extend(existed_train_datalist)
                    concrete_dataset.data_path_list.extend(label_local_path_list)
                    concrete_dataset.data_path_list.extend(pseudo_local_path_list)
                    break
            else:
                # 默认复杂训练方式（如，半监督训练，跨域训练模式）
                # 仅支持list size=2 （默认，0: 标签数据+伪标签数据；1: 无标签数据）
                concrete_dataset = cfg.data.train[0]
                while True:
                    if concrete_dataset.type in control_dataset_type_names:
                        concrete_dataset = concrete_dataset.dataset
                        continue
                    concrete_dataset.data_path_list.extend(existed_train_datalist)
                    concrete_dataset.data_path_list.extend(label_local_path_list)
                    concrete_dataset.data_path_list.extend(pseudo_local_path_list)
                    break

                concrete_dataset = cfg.data.train[1]
                while True:
                    if concrete_dataset.type in control_dataset_type_names:
                        concrete_dataset = concrete_dataset.dataset
                        continue
                    # 无标签数据
                    concrete_dataset.data_path_list.extend(unlabel_local_path_list)
                    # 跨域数据
                    concrete_dataset.data_path_list.extend(cross_domain_local_path_list)
                    break

            # 测试集相关 (仅存在无标签数据时，默认是主动学习过程)
            if nn_args.process == 'activelearning':
                assert(len(unlabel_local_path_list) > 0)
                cfg.data.test.data_path_list = unlabel_local_path_list

            # step3.4: 模型配置扩展
            if 'model' in extra_config:
                # 更新默认配置
                default_model_cfg = cfg.model
                if 'model' in extra_config['model'] and \
                    'teacher' in extra_config['model']['model']:
                    cfg.model = dict(
                        type=extra_config['model']['type'],
                        model=dict(
                            teacher=extra_config['model']['model']['teacher'],
                            student=default_model_cfg
                        ),
                        train_cfg=extra_config['model']['train_cfg'] if 'train_cfg' in extra_config['model'] else None,
                        test_cfg=extra_config['model']['test_cfg'] if 'test_cfg' in extra_config['model'] else None,
                        init_cfg=extra_config['model']['init_cfg'] if 'init_cfg' in extra_config['model'] else None
                    )
                    
                    if cfg.model.model.teacher is None:
                        cfg.model.model.teacher = default_model_cfg
                    if cfg.model.model.student is None:
                        cfg.model.model.student = default_model_cfg
                        
                    if cfg.model.train_cfg is not None:
                        if 'student' in cfg.model.train_cfg:
                            align_name = cfg.model.train_cfg.align
                            if 'channels' in cfg.model.train_cfg.student:
                                cfg.model.train_cfg.student.channels = cfg.info.get(align_name).channels
                            if 'shapes' in cfg.model.train_cfg.student:
                                cfg.model.train_cfg.student.shapes = cfg.info.get(align_name).shapes 
                else:
                    cfg.model = dict(
                        type=extra_config['model']['type'],
                        model=default_model_cfg,
                        train_cfg=extra_config['model']['train_cfg'] if 'train_cfg' in extra_config['model'] else None,
                        test_cfg=extra_config['model']['test_cfg'] if 'test_cfg' in extra_config['model'] else None,
                        init_cfg=extra_config['model']['init_cfg'] if 'init_cfg' in extra_config['model'] else None
                    )

            # step3.5: 模型运行时hooks扩展
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
            
            # step3.6: 模型其他方面扩展
            if 'optimizer' in extra_config:
                cfg.optimizer = extra_config['optimizer']
            if 'optimizer_config' in extra_config:
                cfg.optimizer_config = extra_config['optimizer_config']
            if 'lr_config' in extra_config:
                cfg.lr_config = extra_config['lr_config']
            if 'max_epochs' in extra_config:
                cfg.max_epochs = extra_config['max_epochs']

    # step4 检查checkpoint
    if nn_args.checkpoint is not None and nn_args.checkpoint != '':
        if not nn_args.checkpoint.startswith('/') and not nn_args.checkpoint.startswith('./'):
            checkpoint_file_name = nn_args.checkpoint.split('/')[-1]
            nn_args.checkpoint = os.path.join('checkpoint-storage', checkpoint_file_name)
            if not os.path.exists(nn_args.checkpoint):
                logging.error(f'Checkpoint {nn_args.checkpoint} not in local.')
    if nn_args.resume_from is not None and nn_args.resume_from != '':
        if not nn_args.resume_from.startswith('/') and not nn_args.resume_from.startswith('./'):
            checkpoint_file_name = nn_args.resume_from.split('/')[-1]
            nn_args.resume_from = os.path.join('checkpoint-storage', checkpoint_file_name)
            if not os.path.exists(nn_args.resume_from):
                logging.error(f'Checkpoint {nn_args.resume_from} not in local.')

    # step5 添加root (运行时，输出结果保存的根目录地址)
    cfg.root = nn_args.root if nn_args.root != '' else './output/'
    cfg.root = os.path.join(cfg.root, nn_args.exp)
    # step5.1 添加root地址（影响checkpoint_config, evaluation）
    if cfg.root != '':
        cfg.checkpoint_config.out_dir = cfg.root
        cfg.evaluation.out_dir = cfg.root
    
    # step6: 执行指令(训练、测试、模型导出)
    if nn_args.process == 'train':
        # 创建训练过程
        trainer = Trainer(
            cfg, 
            './', 
            int(nn_args.gpu_id), # 对于多卡运行环境,会自动忽略此参数数值
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
            int(nn_args.gpu_id), # 对于多卡运行环境,会自动忽略此参数数值
            distributed=nn_args.distributed)
        tester.config_model(checkpoint=nn_args.checkpoint)
        tester.evaluate()
    elif nn_args.process == 'activelearning':
        # 创建主动学习过程,挑选等待标注样本
        print(f'nn_args.distributed {nn_args.distributed}')
        print(f'nn_args.gpu-id {nn_args.gpu_id}')
        print(f'nn_args.checkpoint {nn_args.checkpoint}')        
        ac = Activelearning(cfg, './', int(nn_args.gpu_id), distributed=nn_args.distributed)
        ac.config_model(checkpoint=nn_args.checkpoint, revise_keys=[('^','model.')])
        ac.select(nn_args.exp)
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