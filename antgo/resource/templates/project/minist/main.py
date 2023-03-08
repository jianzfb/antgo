# (1) 训练过程
# 单机4卡训练
# antgo train main.py 4 --exp=task --no-validate --config=./config.py
# 单机1卡训练（可以自定义使用第几块卡 --gpu_id=0）
# antgo train main.py 1 --gpu_id=0 --exp=task --no-validate --config=./config.py
# 单机CPU训练（仅用于调试）
# python3 main.py --exp=task --gpu_id=-1 --no-validate --process=train --config=./config.py

# (2) 评估过程
# 单机4卡评估
# antgo test main.py 4 --exp=task --config=./config.py
# 单机1卡评估（可以自定义使用第几块卡 --gpu_id=0）
# antgo test main.py 1 --gpu_id=0 --exp=task --config=./config.py
# 单机CPU测试（仅用于调试）
# python3 main.py --exp=task --gpu_id=-1 --process=test -config=./config.py

# (3) 模型导出过程
# antgo export main.py --exp=task --checkpoint='' --config=./config.py

from concurrent.futures import process
import sys
import os
import argparse
import torch

# 导入扩展模块 (包括自定义的数据集，模型，等)
from ext import *

# 导入相关包
from antgo.utils import args
from antgo.framework.helper.trainer import *
from antgo.framework.helper.tester import *
from antgo.framework.helper.exporter import *
from antgo.framework.helper.models.detectors import *
from antgo.framework.helper.dataset.pipelines import *
from antgo.framework.helper.utils import Config

# 定义shell参数
# (1) 自定义扩展参数
# 例子：
# args.DEFINE_string('root_hdfs', '', 'label dataset hdfs')

# (2) 定义标准nn参数
args.DEFINE_nn_args()

def main():
    # step1: 加载参数
    nn_args = args.parse_args()
    args.print_args(nn_args)
    assert(nn_args.exp != '')
    print('1')
    # step2: 加载配置文件
    cfg = Config.fromfile(os.path.join(nn_args.exp, nn_args.config))

    print('2')
    # step3: 执行指令（训练、测试、模型导出）
    if nn_args.process == 'train':
        print('3')
        # 创建训练过程
        trainer = Trainer(
            cfg, 
            './', 
            nn_args.gpu_id, # 对于多卡运行环境，会自动忽略此参数数值
            distributed=nn_args.distributed, 
            diff_seed=nn_args.diff_seed, 
            deterministic=nn_args.deterministic, 
            find_unused_parameters=nn_args.find_unused_parameters)
        print('1')
        trainer.config_dataloader(with_validate=not nn_args.no_validate)
        print('5')
        trainer.config_model(resume_from=nn_args.resume_from, load_from=nn_args.checkpoint)
        print('6') 
        if nn_args.max_epochs < 0:
            # 优先使用外部指定max_epochs
            nn_args.max_epochs = cfg.max_epochs
        print(f'max epochs {nn_args.max_epochs}')

        trainer.start_train(max_epochs=nn_args.max_epochs)
        print('7')
    elif nn_args.process == 'test':
        # 创建测试过程
        print(f'nn_args.distributed {nn_args.distributed}')
        print(f'nn_args.gpu_id {nn_args.gpu_id}')
        print(f'nn_args.checkpoint {nn_args.checkpoint}')
        tester = Tester(
            cfg, 
            './',
            nn_args.gpu_id, # 对于多卡运行环境，会自动忽略此参数数值
            distributed=nn_args.distributed)
        tester.config_model(checkpoint=nn_args.checkpoint)
        tester.evaluate()
    elif nn_args.process == 'export':
        # 创建导出模型过程
        tester = Exporter(cfg, './')
        tester.export(
            input_tensor_list=[torch.zeros(shape, dtype=torch.float32) for shape in cfg.export.input_shape_list], 
            input_name_list=cfg.export.input_name_list, 
            output_name_list=cfg.export.output_name_list, 
            checkpoint=nn_args.checkpoint, 
            prefix=f'{nn_args.exp}-model')

if __name__ == "__main__":
    main()