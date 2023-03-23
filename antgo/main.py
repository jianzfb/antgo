# encoding=utf-8
# @Time    : 17-3-3
# @File    : main.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from inspect import trace
from random import shuffle
import sys
import os
from antgo.utils.utils import *
from antgo.utils.args import *
from antgo.ant.utils import *
from antgo.command import *
from antgo.framework.helper.tools.util import *
from antgo.help import *
from antgo import config
from antgo import version
from jinja2 import Environment, FileSystemLoader
import json
from antgo import tools
from antgo.script import *
import traceback
import yaml

# 需要使用python3
assert(int(sys.version[0]) >= 3)


#############################################
#######   antgo parameters            #######
#############################################
DEFINE_string('project', '', 'set project name')
DEFINE_string('git', None, '')
DEFINE_string('branch', None, '')
DEFINE_string('commit', None, '')
DEFINE_string('image', '', '')      # 镜像
DEFINE_string('gpu-ids', '0', 'use gpu ids')
DEFINE_int('cpu', 0, 'set cpu number')
DEFINE_int('memory', 0, 'set memory size (M)')
DEFINE_string('name', None, '')     # 名字
DEFINE_string('type', None, '')     # 类型
DEFINE_indicator('cloud', True, '') # 指定云端运行


############## submitter ###################
DEFINE_indicator('ssh', True, '')     # ssh 提交
DEFINE_indicator('k8s', True, '')     # k8s 提交
DEFINE_indicator('custom', True, '')  # 开发者自定义任务提交配置

############## global config ###############
DEFINE_string('token', None, '')
############## project config ##############
DEFINE_string('semi', "", 'set semi supervised method')
DEFINE_string("distillation", "", "set distillation method")
DEFINE_string("activelearning", "", "set activelearning method")
DEFINE_string("ensemble", "", "set ensemble method")

############## tool config ##############
DEFINE_string("src", "./", "set src folder/file")
DEFINE_string("tgt", "./", "set tgt folder/file")
DEFINE_int("frame-rate", 30, "video frame rate")
DEFINE_string("prefix", None, "filter by prefixe")
DEFINE_string("suffix", None, "filter by suffix")
DEFINE_string("ext", None, "filter by ext")
DEFINE_string("white-users", None, "name:password,name:password")
DEFINE_string("tags", None, "tag info")
DEFINE_string("no-tags", None, "tag info")
DEFINE_indicator("feedback", True, "")
DEFINE_indicator("user-input", True, "")
DEFINE_int('num', 0, "number")
DEFINE_indicator("to", True, "")
DEFINE_indicator("from", True, "")
DEFINE_indicator("extra", True, "load extra package(mano,smpl,...)")
DEFINE_indicator("shuffle", True, "")
DEFINE_int("max-size", 0, "")
DEFINE_indicator("ignore-incomplete", True, "")

#############################################
DEFINE_nn_args()

action_level_1 = ['train', 'eval', 'export', 'config', 'submitter']
action_level_2 = ['add', 'del', 'create', 'register','update', 'show', 'get', 'tool']


def main():
  main_logo()
  # 备份脚本参数
  sys_argv_cp = sys.argv
  
  # 解析参数
  action_name = sys.argv[1]
  sub_action_name = None
  if action_name in action_level_1:
    sys.argv = [sys.argv[0]] + sys.argv[2:]
  else:
    sub_action_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
  args = parse_args()

  ######################################### 配置文件操作 ################################################
  # 检查配置文件是否存在
  if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
    # 使用指定配置文件更新
    if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
      os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

    config_data = {'FACTORY': './.factory', 'USER_TOKEN': ''}
    env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
    config_template = env.get_template('config.xml')
    config_content = config_template.render(**config_data)

    with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'), 'w') as fp:
      fp.write(config_content)
    logging.warn('Using default config file.')
  
  # 配置操作
  if action_name == 'config':
    config_data = {'FACTORY': '', 'USER_TOKEN': ''}
    # 读取现有数值
    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    config.AntConfig.parse_xml(config_xml)
    config_data['FACTORY'] = getattr(config.AntConfig, 'factory', '')
    config_data['USER_TOKEN'] = getattr(config.AntConfig, 'token', '')
    
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

  if not os.path.exists(config.AntConfig.factory):
    os.makedirs(config.AntConfig.factory)
  if not os.path.exists(config.AntConfig.data_factory):
    os.makedirs(config.AntConfig.data_factory)
  if not os.path.exists(config.AntConfig.task_factory):
    os.makedirs(config.AntConfig.task_factory)

  ######################################### 支持扩展 ###############################################
  if args.extra and not os.path.exists('extra'):
    logging.info('download extra package')
    os.system('wget http://image.mltalker.com/extra.tar; tar -xf extra.tar; cd extra/manopth; python3 setup.py install')
  
  if args.ext_module != '':
    logging.info('import extent module')
    load_extmodule(args.ext_module)

  ##################################### 支持任务提交脚本配置  ###########################################
  if action_name == 'submitter':
    if args.ssh:
      # ssh提交配置
      if args.config == '' or args.config == 'config.py':
        # 生成ssh提交配置模板
        ssh_submit_config_file = os.path.join(os.path.dirname(__file__), 'script', 'ssh-submit-config.yaml')
        shutil.copy(ssh_submit_config_file, './')
        return
      else:
        # 更新ssh提交配置模板
        assert(os.path.exists(args.config))
        shutil.copy(args.config, os.path.join(os.environ['HOME'], '.config', 'antgo'))
        return 

  ######################################### 生成最小mvp ###############################################
  if action_name == 'create' and sub_action_name == 'mvp':
    project_folder = os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'mvp')
    generate_project_exp_example(project_folder, args.tgt)
    return

  ####################################################################################################
  # 执行指令
  if action_name in action_level_1:
    # 云端任务执行
    if args.cloud:
      # 检查项目环境
      if not check_project_environment(args):
        return
            
      # 记录commit
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp)

      if args.exp not in project_info['exp']:
        logging.error(f'Exp {args.exp} not exist in project {args.project}.')
        return

      try:
        rep = git.Repo('./')     
        project_info['exp'][args.exp]['branch'] = rep.active_branch.name
        project_info['exp'][args.exp]['commit'] = rep.active_branch.commit.name_rev          
      except Exception:
          pass
      
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)
      
      # 云端提交        
      sys_argv_new = sys_argv_cp.replace('--cloud', '')        
      # step 1.1: 检查提交脚本配置
      if args.ssh:
        ssh_submit_process_func(args.project, sys_argv_new, 0 if args.gpu_ids == '' else len(args.gpu_ids.split(',')), args.cpu, args.memory)  
      return

    # 本地任务执行
    if action_name == 'train':
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
        command_str = f'bash install.sh; bash launch.sh {args.exp}/main.py {gpu_num} --exp={args.exp}  --process=train'
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
        command_str = f'bash install.sh; bash launch.sh {args.exp}/main.py {gpu_num} --exp={args.exp} --process=test'
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
        with open(os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project.json'), 'r') as fp:
          project_config = json.load(fp)

        # 配置定制化项目
        project_config['name'] = args.name
        project_config['type'] = args.type
        project_config['git'] = args.git
        project_config['image'] = args.image
        # 设置自动优化工具
        project_config['tool']['semi']['method'] = args.semi
        project_config['tool']['distillation']['method'] = args.distillation
        project_config['tool']['activelearning']['method'] = args.activelearning
        project_config['tool']['ensemble']['method'] = args.ensemble

        # 在本地存储项目信息
        with open(os.path.join(config.AntConfig.task_factory,f'{args.name}.json'), 'w') as fp:
          json.dump(project_config, fp)
      elif sub_action_name == 'exp':
        # 检查项目环境
        if not check_project_environment(args):
          return

        # 获得基本项目信息
        if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
          logging.error(f'Project {args.project} has not been created.')
          return
        project_info = {}
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
          project_info = json.load(fp) 

        # 项目必须存在git
        assert(project_info['git'] != '')    
        
        # 创建项目实验
        assert(args.name is not None)
        if args.name in project_info['exp']:
          logging.error(f'Experiment {args.name} existed')
          return        
        
        if not os.path.exists(args.name):
          # 如果不存在实验目录，创建
          os.makedirs(args.name)

        project_info['exp'][args.name] = {
          "exp":"",
          "branch":"",
          "commit":"",
          "web":"",
          "metric":{},
          "dataset": {"test": "", "train": []},
          "checkpoint": "",
          "create_time": "",
          "finish_time": ""
        }
        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
          json.dump(project_info, fp)
    elif action_name == 'tool':
      # 工具相关
      if sub_action_name.startswith('extract'):
        # extract/videos, extract/images
        tool_func = getattr(tools, f'extract_from_{sub_action_name.split("/")[1]}', None)
        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        tool_func(
          args.src, 
          args.tgt, 
          frame_rate=args.frame_rate, 
          filter_prefix=args.prefix, 
          filter_suffix=args.suffix, 
          filter_ext=args.ext,
          feedback=args.feedback,
          num=args.num,
          shuffle=args.shuffle,
          max_size=args.max_size)
      elif sub_action_name.startswith('browser'):
        tool_func = getattr(tools, f'browser_{sub_action_name.split("/")[-1]}', None)
        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        tool_func(args.src, args.tags, args.white_users, args.feedback, args.user_input)
      elif sub_action_name.startswith('filter'):
        tool_func = getattr(tools, f'filter_by_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return 
        
        tool_func(args.src, args.tgt, args.tags, args.no_tags)
        return
      elif sub_action_name.startswith('package'):
        tool_func = getattr(tools, f'package_to_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        
        # src: json文件路径
        # tgt: 打包数据存放目录
        # prefix: 打包数据的文件前缀
        # num: 每个shard内样本条数
        tool_func(args.src, args.tgt, args.prefix, args.num)
      elif sub_action_name.startswith('download'):
        tool_func = getattr(tools, f'download_from_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return
        
        tool_func(args.tgt, args.tags)
      elif sub_action_name.startswith('label'):
        if sub_action_name.split("/")[1] == 'start':
          tool_func = getattr(tools, f'label_{sub_action_name.split("/")[1]}', None)
          if tool_func is None:
            logging.error(f'Tool {sub_action_name} not exist.')
            return
          
          tool_func(args.src, args.tgt, args.tags, args.type, args.white_users, ignore_incomplete=args.ignore_incomplete)
          return
        else:
          tool_func = None
          if args.to:
            tool_func = getattr(tools, f'label_to_{sub_action_name.split("/")[1]}', None)
          else:
            tool_func = getattr(tools, f'label_from_{sub_action_name.split("/")[1]}', None)
            
          if tool_func is None:
            logging.error(f'Tool {sub_action_name} not exist.')
            return
          
          tool_func(args.src, args.tgt, prefix=args.prefix, tags=args.tags, ignore_incomplete=args.ignore_incomplete)

      
if __name__ == '__main__':
  main()
