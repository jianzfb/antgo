# encoding=utf-8
# @Time    : 17-3-3
# @File    : main.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
from antgo.utils.utils import *
from antgo.utils.args import *
from antgo.ant.utils import *
from antgo.ant.client import get_client, launch_server
from antgo.command import *
from antgo.framework.helper.tools.util import *
from antgo.help import *
from antgo import config
from jinja2 import Environment, FileSystemLoader
import json
from antgo import tools
from antgo.script import *
import yaml
from pathlib import Path
from aligo import Aligo


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
DEFINE_int('cpu', 0, 'set cpu number')          # cpu 数
DEFINE_int('gpu', 0, 'set gpu number')          # gpu 数
DEFINE_int('memory', 0, 'set memory size (M)')  # 内存大小（单位M）
DEFINE_string('name', None, '')     # 名字
DEFINE_string('type', None, '')     # 类型
DEFINE_string("address", None, "")
DEFINE_indicator("auto", True, '')  # 是否项目自动优化
DEFINE_indicator("finetune", True, '')  # 是否启用finetune模式
DEFINE_string('id', None, '')
DEFINE_string("ip", "", "set ip")
DEFINE_int('port', 0, 'set port')
DEFINE_choices('stage', 'supervised', ['supervised', 'semi-supervised', 'distillation', 'activelearning', 'label'], '')

############## submitter ###################
DEFINE_indicator('ssh', True, '')     # ssh 提交
DEFINE_indicator('k8s', True, '')     # k8s 提交
DEFINE_indicator('local', True, '')   # 本地提交

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
DEFINE_float("ext-ratio", 0.35, "")
DEFINE_int("ext-size", 0, "")
DEFINE_indicator("ignore-incomplete", True, "")

#############################################
DEFINE_nn_args()

action_level_1 = ['train', 'eval', 'export', 'config', 'server', 'activelearning', 'device', 'stop', 'ls', 'log']
action_level_2 = ['add', 'del', 'create', 'register','update', 'show', 'get', 'tool', 'share', 'download', 'upload', 'submitter']


def main():
  # 显示antgo logo
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

  # 查看运行设备（本地/远程）
  if action_name == 'device':
    tools.check_device_info(args)
    return

  # 操作执行任务（本地/远程）
  if action_name in ['stop', 'ls', 'log']:
    tools.operate_on_running_status(action_name, args)
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
    if sub_action_name is None or sub_action_name == '':
      logging.error(f'Only support {action_name} activate/ls/update/template')
      return
    if sub_action_name == 'template':
      # 生成任务提交配置
      if args.ssh:
        # 生成ssh提交配置模板
        ssh_submit_config_file = os.path.join(os.path.dirname(__file__), 'script', 'ssh-submit-config.yaml')
        shutil.copy(ssh_submit_config_file, './')        
      else:
        # 生成自定义的提交配置模板
        submit_config_file = os.path.join(os.path.dirname(__file__), 'script', 'submit-config.yaml')
        shutil.copy(submit_config_file, './')
    elif sub_action_name == 'update':
      # 更新任务提交配置
      has_config_file = not (args.config == '' or args.config == 'config.py')
      has_config_file = has_config_file and os.path.exists(args.config)
      if not has_config_file:
        logging.error('Need set --config=')
        return

      if args.ssh:
        # 检查是否支持免密登录
        if not os.path.exists(os.path.join(os.environ['HOME'], '.ssh', 'id_rsa')):
          os.system(f'cd {os.path.join(os.environ["HOME"], ".ssh")} && ssh-keygen -t rsa')

        # 远程执行命令
        with open(args.config, 'r') as fp:
          ssh_config_info = yaml.safe_load(fp)

        shutil.copy(args.config, os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml'))
        shutil.copy(args.config, os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{ssh_config_info["config"]["ip"]}-submit-config.yaml'))

        shutil.copy(os.path.join(os.path.dirname(__file__), 'script', 'ssh_nopassword_config.sh'), os.path.join(os.environ["HOME"], ".ssh", 'user_ssh_nopassword_config.sh'))
        os.system(f'cd {os.path.join(os.environ["HOME"], ".ssh")} && rsa=`cat id_rsa.pub` && ' + "sed -i 's%placeholder%'"+"\"${rsa}\""+"'%g' user_ssh_nopassword_config.sh")
        os.system(f'cd {os.path.join(os.environ["HOME"], ".ssh")} && ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} < user_ssh_nopassword_config.sh')
      else:
        logging.info("Only support ssh remote task submitter")
        return

      print(f'update submitter config {args.config}')
    elif sub_action_name == 'ls':
      # 输出已经配置的远程信息
      if args.ssh:
        for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
          if file_name.endswith('.yaml'):
            terms = file_name.split('-')
            if len(terms) == 4:
              pprint(f'{terms[1]}')
      else:
        logging.info("Only support ssh remote task submitter")
    elif sub_action_name == 'activate':
      # 将选定远程作为默认
      if args.ssh:
        if args.ip == '':
          logging.error('Need set --ip=')
          return
        if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{args.ip}-submit-config.yaml')):
          logging.error(f'Dont exist ssh-{args.ip}-submit-config.yaml')
          return

        shutil.copy(
          os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{args.ip}-submit-config.yaml'),
          os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
        )
    else:
      logging.error("Only support ssh remote task submitter")
      return

    return

  ######################################### 后台监控服务 ################################################
  if action_name == 'server':
    os.system(f'nohup python3 {os.path.join(os.path.dirname(__file__), "ant", "client.py")} --port={args.port} --root={args.root} --ext-module={args.ext_module} > /tmp/antgo.server.log 2>&1 &')
    return
    # launch_server(args.port, args.root, args.ext_module)
    # return

  # 检查是否后台服务活跃
  if not get_client().alive():
    logging.warn('Antgo auto server not work. Please run "antgo server" to launch.')

  ######################################### 生成最小mvp ###############################################
  if action_name == 'create' and sub_action_name == 'mvp':
    # 在项目目录下生成mvp
    project_folder = os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'mvp')
    generate_project_exp_example(project_folder, args.tgt, args.name)
    return

  ####################################################################################################
  # 执行指令
  if action_name in action_level_1:
    if args.root is None or args.root == '':
      print('Using default root address ali:///exp')
      args.root = "ali:///exp"

    if args.root.startswith('ali:'):
      # 尝试进行认证，从而保证当前路径下生成认证信息
      # do nothing
      if not os.path.exists('./aligo.json'):
        ali = Aligo()
        shutil.copy(os.path.join(Path.home().joinpath('.aligo'), 'aligo.json'),'./')

    # 创建root根目录
    file_client_mkdir(args.root)

    # 远程提交任务(local模式仅在开发者调试时使用)
    if args.ssh or args.k8s or args.local:
      # 允许非标准项目远程提交
      if args.project == '':
        # 非标准项目模式
        filter_sys_argv_cp = []
        for t in sys_argv_cp:
          if t.startswith('--project'):
            continue
          if t.startswith('--root'):
            continue
          filter_sys_argv_cp.append(t)

        sys_argv_cp = filter_sys_argv_cp
        sys_argv_cp.append(f'--root={args.root}')
        sys_argv_cmd = ' '.join(sys_argv_cp[1:])

        # 直接进行任务提交
        # step 1.1: 检查提交脚本配置
        if args.local:
          # 本地提交
          sys_argv_cmd = sys_argv_cmd.replace('--local', '')
          sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
          sys_argv_cmd = f'antgo {sys_argv_cmd}'
          local_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)  
        elif args.ssh:
          # ssh提交
          sys_argv_cmd = sys_argv_cmd.replace('--ssh', '')
          sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
          sys_argv_cmd = f'antgo {sys_argv_cmd}'        
          ssh_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)  
        else:
          # 自定义脚本提交
          sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
          sys_argv_cmd = f'antgo {sys_argv_cmd}'          
          custom_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)
        return

      # 标准项目模式（需要检查项目规范性）
      # 检查项目环境
      if not check_project_environment(args):
        return

      # 检查项目和实验是否存在
      if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} hasnot create.')
        return

      project_info = {}      
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp)

      if args.exp not in project_info['exp']:
        logging.error(f'Exp {args.exp} not exist in project {args.project}.')
        return

      if not os.path.exists(args.exp):
        logging.error(f'Exp code not found in current folder.')
        return

      # 检查任务提交功能
      if args.ssh:
        ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
        if not os.path.exists(ssh_submit_config_file):
          logging.error('No ssh submit config.')
          return
      elif not args.local:
        submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'submit-config.yaml')    
        if not os.path.exists(submit_config_file):
            logging.error('No custom submit script config')
            return

      # 记录commit
      rep = git.Repo('./')     
      if not isinstance(project_info['exp'][args.exp], list):
        project_info['exp'][args.exp] = []
      
      exp_info = exp_basic_info()
      exp_info['exp'] = args.exp
      exp_info['id'] = time.strftime('%Y-%m-%dx%H-%M-%S',time.localtime(time.time()))
      exp_info['branch'] = rep.active_branch.name
      exp_info['commit'] = rep.active_branch.commit.name_rev
      exp_info['state'] = 'running'
      exp_info['stage'] = args.stage
      project_info['exp'][args.exp].append(
        exp_info
      )

      # 创建项目文件
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)

      # 创建脚本命令
      sys_argv_cp.append(f'--id={exp_info["id"]}')
      filter_sys_argv_cp = []
      for t in sys_argv_cp:
        if t.startswith('--project'):
          continue
        if t.startswith('--root'):
          continue
        filter_sys_argv_cp.append(t)

      sys_argv_cp = filter_sys_argv_cp
      sys_argv_cp.append(f'--root={args.root}/{args.project}')
      sys_argv_cmd = ' '.join(sys_argv_cp[1:])      

      # step 1.1: 检查提交脚本配置
      if args.local:
        # 本地提交
        sys_argv_cmd = sys_argv_cmd.replace('--local', '')
        sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
        sys_argv_cmd = f'antgo {sys_argv_cmd}'
        local_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)  
      elif args.ssh:
        # ssh提交
        sys_argv_cmd = sys_argv_cmd.replace('--ssh', '')
        sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
        sys_argv_cmd = f'antgo {sys_argv_cmd}'        
        ssh_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)  
      else:
        # 自定义脚本提交
        sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
        sys_argv_cmd = f'antgo {sys_argv_cmd}'          
        custom_submit_process_func(args.project, sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory)
      return

    # 本地任务执行
    elif args.project != '':
      # 场景：本地直接任务执行
      # 标准项目需要检查任务规范性
      # 检查项目环境
      if not check_project_environment(args):
        return

      # 检查项目和实验是否存在
      if not os.path.exists(os.path.join(config.AntConfig.task_factory,f'{args.project}.json')):
        logging.error(f'Project {args.project} hasnot create.')
        return

      project_info = {}      
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'r') as fp:
        project_info = json.load(fp)

      if args.exp not in project_info['exp']:
        logging.error(f'Exp {args.exp} not exist in project {args.project}.')
        return

      if not os.path.exists(args.exp):
        logging.error(f'Exp code not found in current folder.')
        return

      # 记录commit
      rep = git.Repo('./')             
      if not isinstance(project_info['exp'][args.exp], list):
        project_info['exp'][args.exp] = []

      exp_info = exp_basic_info()
      exp_info['exp'] = args.exp
      exp_info['id'] = time.strftime('%Y-%m-%dx%H-%M-%S',time.localtime(time.time()))
      exp_info['branch'] = rep.active_branch.name
      exp_info['commit'] = rep.active_branch.commit.name_rev
      exp_info['state'] = 'running'
      exp_info['stage'] = args.stage
      project_info['exp'][args.exp].append(
        exp_info
      )

      # 创建项目文件
      with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
        json.dump(project_info, fp)

      # exp, exp:id
      args.id = exp_info['id']

    # 检查后端存储功能
    if args.root is not None and args.root != '':
      status = FileClient.infer_client(uri=args.root)
      if status is None:
        logging.error(f"Fail check backend storage {args.root}")

    # 执行任务
    auto_exp_name = f'{args.exp}.{args.id}' if args.id is not None else args.exp
    script_folder = os.path.join(os.path.dirname(__file__), 'script')
    if action_name == 'train':
      if args.checkpoint is None:
        args.checkpoint = ''
      if args.resume_from is None:
        args.resume_from = ''

      if args.gpu_id == '' or int(args.gpu_id.split(',')[0]) == -1:
        # cpu run
        # (1)安装;(2)数据准备;(3)运行
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint} --resume-from={args.resume_from}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={-1} --process=train --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'

        os.system(command_str)
      elif len(args.gpu_id.split(',')) == 1:
        # single gpu run
        # (1)安装;(2)数据准备;(3)运行
        gpu_id = args.gpu_id.split(',')[0]
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint} --resume-from={args.resume_from}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={gpu_id} --process=train --root={args.root} --extra-config={args.extra_config} --config={args.config}'
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
        # (1)安装;(2)数据准备;(3)运行
        gpu_num = len(args.gpu_id.split(','))
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint} --resume-from={args.resume_from}; bash launch.sh {args.exp}/main.py {gpu_num} --exp={auto_exp_name}  --process=train --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'

        os.system(command_str)
    elif action_name == 'activelearning':
      if args.checkpoint is None:
        args.checkpoint = ''

      if args.gpu_id == '' or int(args.gpu_id.split(',')[0]) == -1:
        # cpu run
        # (1)安装;(2)数据准备;(3)运行
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={-1} --process=activelearning --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'

        os.system(command_str)
      elif len(args.gpu_id.split(',')) == 1:
        # single gpu run
        # (1)安装;(2)数据准备;(3)运行
        gpu_id = args.gpu_id.split(',')[0]
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={gpu_id} --process=activelearning --root={args.root} --extra-config={args.extra_config} --config={args.config}'
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
        # (1)安装;(2)数据准备;(3)运行
        gpu_num = len(args.gpu_id.split(','))
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; bash launch.sh {args.exp}/main.py {gpu_num} --exp={auto_exp_name}  --process=activelearning --root={args.root} --extra-config={args.extra_config} --config={args.config}'
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
      if args.checkpoint is None:
        logging.error('Must set --checkpoint=')
        return

      # (1)安装;(2)数据准备;(3)运行
      if args.gpu_id == '' or int(args.gpu_id.split(',')[0]) == -1:
        # cpu run
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={-1} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      elif len(args.gpu_id.split(',')) == 1:
        # single gpu run
        gpu_id = args.gpu_id.split(',')[0]
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={gpu_id} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      else:
        # multi gpu run
        gpu_num = len(args.gpu_id.split(','))
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; bash launch.sh {args.exp}/main.py {gpu_num} --exp={auto_exp_name} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
    elif action_name == 'export':
      if args.checkpoint is None:
        logging.error('Must set --checkpoint=')
        return
      os.system(f'bash install.sh; python3 {args.exp}/main.py --exp={auto_exp_name} --checkpoint={args.checkpoint} --process=export --root={args.root} --config={args.config}')
  else:
    if action_name == 'create':
      if sub_action_name == 'project':
        if args.git is None or args.git == '':
          logging.error('Must set project git with --git')
          return

        # 自动提取git的后缀作为项目名称
        project_name = args.git.split('/')[-1].split('.')[0]

        if os.path.exists(os.path.join(config.AntConfig.task_factory,f'{project_name}.json')):
          logging.error(f'Project {project_name} has existed.')
          return

        # 加载项目默认模板
        with open(os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project.json'), 'r') as fp:
          project_config = json.load(fp)

        # 配置定制化项目
        project_config['name'] = project_name    # 项目名称
        project_config['type'] = args.type if args.type is not None else '' # 项目类型
        project_config['git'] = args.git      # 项目git地址
        project_config['image'] = args.image  # 项目需要的定制化的镜像
        project_config['auto'] = args.auto    # 项目是否进行自动化更新  （半监督\蒸馏\迁移）
        # 设置自动优化工具
        project_config['tool']['semi']['method'] = args.semi
        project_config['tool']['distillation']['method'] = args.distillation
        project_config['tool']['activelearning']['method'] = args.activelearning
        project_config['tool']['ensemble']['method'] = args.ensemble

        # 在本地存储项目信息
        with open(os.path.join(config.AntConfig.task_factory,f'{project_name}.json'), 'w') as fp:
          json.dump(project_config, fp)

        # 准备项目代码
        prepare_project_environment(args.git, args.branch, args.commit)
        pprint(project_config)
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

        if args.name not in project_info['exp']:
          project_info['exp'][args.name] = []

        with open(os.path.join(config.AntConfig.task_factory,f'{args.project}.json'), 'w') as fp:
          json.dump(project_info, fp)
    elif action_name == 'add':
      # expert, product, basline, train/label, train/unlabel, test, val
      project_add_action(sub_action_name, args)
    elif action_name == 'del':
      # expert, product, basline, train/label, train/unlabel, test, val
      project_del_action(sub_action_name, args)
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
          max_size=args.max_size,
          ext_ratio=args.ext_ratio)
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

        # args.src 远程文件路径
        # args.tgt 本地路径
        # args.tags 关键字（对于搜索引擎下载使用）
        tool_func(args.tgt, args.tags, src_path=args.src)
      elif sub_action_name.startswith('upload'):
        tool_func = getattr(tools, f'upload_to_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return

        # args.src 本地路径
        # args.tgt 远程目录
        tool_func(args.tgt, src_path=args.src)
      elif sub_action_name.startswith('ls'):
        tool_func = getattr(tools, f'ls_from_{sub_action_name.split("/")[1]}', None)

        if tool_func is None:
          logging.error(f'Tool {sub_action_name} not exist.')
          return

        # args.src 远程目录
        tool_func(args.src)
      elif sub_action_name.startswith('label'):
        if sub_action_name.split("/")[1] == 'start':
          tool_func = getattr(tools, f'label_{sub_action_name.split("/")[1]}', None)
          if tool_func is None:
            logging.error(f'Tool {sub_action_name} not exist.')
            return

          tool_func(args.src, args.tgt, args.tags, args.type, args.white_users, ignore_incomplete=args.ignore_incomplete, root=args.root, exp=args.exp)
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
    elif action_name == 'show':
      show_action(sub_action_name, args)
    elif action_name == 'get':
      get_action(sub_action_name, args)
    elif action_name == 'update':
      update_project_config(sub_action_name, args)
    elif action_name == 'share':
      if sub_action_name == 'data':
        tools.share_data_func(args)
    else:
      logging.error(f'Dont support {action_name}')
      logging.info(f'All support action {action_level_1 + action_level_2}')

if __name__ == '__main__':
  main()
