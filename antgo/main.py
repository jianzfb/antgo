# encoding=utf-8
# @Time    : 17-3-3
# @File    : main.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from ast import parse
from random import shuffle
import sys
import os
from typing import NamedTuple
from antgo.utils.utils import *
from antgo.utils.args import *
from antgo.ant.utils import *
from antgo.command import *
from antgo.help import *
from antgo import config
from antgo import version
import traceback
from jinja2 import Environment, FileSystemLoader
import subprocess
import json
import shutil
from antgo import tools

# 需要使用python3
assert(int(sys.version[0]) >= 3)


_ant_support_commands = ["train", "eval", "export"]

#############################################
#######   antgo parameters            #######
#############################################
DEFINE_string('project', '', 'set project name')
DEFINE_string('git', None, '')
DEFINE_string('branch', None, '')
DEFINE_string('commit', None, '')
DEFINE_string('gpu-ids', '0', 'use gpu ids')
DEFINE_int('cpu', 0, 'set cpu number')
DEFINE_int('memory', 0, 'set memory size (M)')
DEFINE_string('name', None, '')   # 名字
DEFINE_string('type', None, '')   # 类型
DEFINE_indicator('with-example', False, 'with example code')
############## global config ###############
DEFINE_string('token', None, '')
DEFINE_string('root', None, '')
############## project config ##############
DEFINE_string('semi', "", 'set semi supervised method')
DEFINE_string("distillation", "", "set distillation method")
DEFINE_string("activelearning", "", "set activelearning method")
DEFINE_string("ensemble", "", "set ensemble method")


############## tool config ##############
DEFINE_string("src", "", "set src folder/file")
DEFINE_string("tgt", "", "set tgt folder/file")
DEFINE_int("frame-rate", 30, "video frame rate")
DEFINE_string("filter-prefix", None, "filter by prefixe")
DEFINE_string("filter-suffix", None, "filter by suffix")
DEFINE_string("filter-ext", None, "filter by ext")
DEFINE_string("white-users", None, "name:password,name:password")
DEFINE_string("tags", None, "tag info")

#############################################
DEFINE_nn_args()

action_level_1 = ['train', 'eval', 'export', 'config']
action_level_2 = ['add', 'del', 'create', 'update', 'show', 'get', 'tool']


def main():
  main_logo()

  # 解析参数
  action_name = sys.argv[1]
  sub_action_name = None
  if action_name in action_level_1:
    sys.argv = [sys.argv[0]] + sys.argv[2:]
  else:
    sub_action_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
  args = parse_args()

  ############################### 配置文件操作 ##########################################################
  # 检查配置文件是否存在
  if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
    # 使用指定配置文件更新
    if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
      os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

    config_data = {'FACTORY': '', 'USER_TOKEN': ''}
    env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
    config_template = env.get_template('config.xml')
    config_content = config_template.render(**config_data)

    with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'), 'w') as fp:
      fp.write(config_content)
    logging.warn('Using default config file.')

  if action_name == 'config':
    config_data = {'FACTORY': '', 'USER_TOKEN': ''}
    if args.root is not None:
      config_data['FACTORY'] = args.root
    if args.token is not None:
      config_data['USER_TOKEN'] = args.token

    env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
    config_template = env.get_template('config.xml')
    config_content = config_template.render(**config_data)

    with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'), 'w') as fp:
      fp.write(config_content)

    logging.info('Update config file.')
    return

  # 解析配置文件
  config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
  config.AntConfig.parse_xml(config_xml)

  factory = getattr(config.AntConfig, 'factory', None)
  if factory is None or \
      factory == '' or not os.path.exists(config.AntConfig.factory):
    logging.error('Factory folder is missing, please run antgo config.')
    return

  if not os.path.exists(config.AntConfig.data_factory):
    os.makedirs(config.AntConfig.data_factory)
  if not os.path.exists(config.AntConfig.task_factory):
    os.makedirs(config.AntConfig.task_factory)

  #################################################################################################
  
  # 执行指令
  if action_name in action_level_1:
    # 检查当前环境
    if not check_project_environment(args):
      return

    if action_name == 'train':
      # step 1: (TODO)在集群提交任务

      # step 2: 本地执行任务
      if args.gpu_ids == '' or int(args.gpu_ids.split(',')[0]) == -1:
        # cpu run
        command_str = f'bash install.sh; python3 {args.exp}/main.py --exp={args.exp} --gpu-id={-1} --process=train'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'
        
        os.system(command_str)
      elif len(args.gpu_ids.split(',')) == 1:
        # single gpu run
        gpu_id = args.gpu_ids.split(',')[0]
        command_str = f'bash install.sh; python3 {args.exp}/main.py --exp={args.exp} --gpu-id={gpu_id} --process=train'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'

        os.system(command_str)
      else:
        # multi gpu run
        gpu_num = len(args.gpu_ids.split(','))
        command_str = f'bash install.sh; bash {args.exp}/dist_run.sh main.py {gpu_num} --exp={args.exp}  --process=train'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'
        
        os.system(command_str)
    elif action_name == 'eval':
      assert(args.checkpoint is not None)
      if args.gpu_ids == '' or int(args.gpu_ids.split(',')[0]) == -1:
        # cpu run
        command_str = f'bash install.sh; python3 {args.exp}/main.py --exp={args.exp} --gpu-id={-1} --process=test'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      elif len(args.gpu_ids.split(',')) == 1:
        # single gpu run
        gpu_id = args.gpu_ids.split(',')[0]
        command_str = f'bash install.sh; python3 {args.exp}/main.py --exp={args.exp} --gpu-id={gpu_id} --process=test'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      else:
        # multi gpu run
        gpu_num = len(args.gpu_ids.split(','))
        command_str = f'bash install.sh; bash dist_run.sh {args.exp}/main.py {gpu_num} --exp={args.exp} --process=test'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
    elif action_name == 'export':
      assert(args.checkpoint is not None)
      os.system(f'bash install.sh; python3 {args.exp}/main.py --exp={args.exp} --checkpoint={args.checkpoint} --process=export')
  else:
    if action_name == 'create':
      if sub_action_name == 'project':
        assert(args.name is not None)
        assert(args.type is not None)
        assert(args.git is not None)
        if os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.name}.json')):
          logging.error(f'Project {args.name} has existed.')
          return

        # 加载项目默认模板
        with open(os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project', 'project.json'), 'r') as fp:
          project_config = json.load(fp)

        # 配置定制化项目
        project_config['name'] = args.name
        project_config['type'] = args.type
        project_config['git'] = args.git
        project_config['semi']['method'] = args.semi
        project_config['distillation']['method'] = args.distillation
        project_config['activelearning']['method'] = args.activelearning
        project_config['ensemble']['method'] = args.ensemble
        
        # TODO, 项目信息保存在云端

        # 在本地存储项目信息
        with open(os.path.join(config.AntConfig.task_factory,f'{args.name}.json'), 'w') as fp:
          json.dump(project_config, fp)
      elif sub_action_name == 'exp':
        # 检查项目环境
        if not check_project_environment(args):
          return

        # 获得基本项目信息
        if os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
          logging.error(f'Project {args.project} has not been created.')
          return
        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
          project_info = json.load(fp) 

        # 项目必须存在git
        assert(project_info['git'] != '')

        # 创建项目实验
        assert(args.name is not None)
        if os.path.exists(args.name):
          logging.error(f'Experiment {args.name} existed')
          return
        os.makedirs(args.name)

        # 生成示例代码
        if args.with_example:
            project_folder = os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project')
            generate_project_exp_example(project_folder, './')
    elif action_name == 'tool':
      # 工具相关
      if sub_action_name.startswith('extract'):
        # extract/videos, extract/images
        tool_func = getattr(tools, f'extract_from_{sub_action_name.split("/")[1]}', None)
        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        tool_func(args.src, args.tgt, frame_rate=args.frame_rate, filter_prefix=args.filter_prefix, filter_suffix=args.filter_suffix, filter_ext=args.filter_ext)
      elif sub_action_name.startswith('browser'):
        tool_func = getattr(tools, f'browser_{sub_action_name.split("/")[-1]}', None)
        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        tool_func(args.src, args.tags, args.white_users)
      elif sub_action_name.startswith('annotation'):
        # annotation/ls, annotation/browser, annotation
        if sub_action_name == 'annotation':
          tool_func = getattr(tools, f'annotation', None)
        else:
          tool_func = getattr(tools, f'annotation_from_{sub_action_name.split("/")[1]}', None)
        
        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        # src: data folder, tgt: annotation file
        tool_func(args.src, args.tgt)
      elif sub_action_name.startswith('filter'):
        tool_func = getattr(tools, f'filter_by_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return 
        
        tool_func(args.src, args.tgt, args.tags)
        return
      elif sub_action_name.startswith('package'):
        # TODO测试
        # src is annotation file
        # tgt is target folder
        tool_func = getattr(tools, f'package_to_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        
        tool_func(args.src, args.tgt)


if __name__ == '__main__':
  main()
