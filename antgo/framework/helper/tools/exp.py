from inspect import trace
import sys
import os
import argparse
from antgo.framework.helper.tools import args
from antgo.framework.helper.utils.config import Config
import pprint
import copy
import traceback


def main(exp_args):
    # 根据传入的参数进行实验规划
    exp_schedule_config = Config.fromfile(exp_args.config)
    # 运行脚本
    exp_launch_script = exp_schedule_config.exp.launch_script

    # 运行参数
    script_args = exp_schedule_config.exp.args

    # 实验规划参数
    exp_num = len(exp_schedule_config.exp.schedule)

    pp = pprint.PrettyPrinter(indent=4)
    for exp_i in range(exp_num):
        # 打印实验进度
        print(f'ANTGO Exp Progress {exp_i+1}/{exp_num}')

        # 打印实验参数设置
        exp_config_info = copy.deepcopy(exp_schedule_config.exp.schedule[exp_i])
        exp_config_info.update({
            'launch_script': f'{exp_launch_script} {script_args}'
        })
        pp.pprint(exp_config_info)

        exp_args = ''
        for exp_arg_k, exp_arg_v in exp_schedule_config.exp.schedule[exp_i].items():
            exp_args += f'{exp_arg_k} {exp_arg_v}'

        exp_launch_cmd = f'{exp_launch_script} {script_args} {exp_args}'

        # 提示实验开始
        try:
            os.system(exp_launch_cmd)
        except Exception as e:
            traceback.print_exc()

    # 提示实验结束
    print('ANTGO Exp Finish')

if __name__ == "__main__":
    exp_args = args.nn_args()
    main(exp_args)