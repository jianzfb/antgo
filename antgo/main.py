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
from antgo.tools.install import *
from antgo.help import *
from antgo import config
from antgo import tools
from antgo.script import *
from jinja2 import Environment, FileSystemLoader
import json
import yaml
from pathlib import Path
from filelock import FileLock
from aligo import Aligo
import antvis.client.mlogger as mlogger
import tempfile


# 需要使用python3
assert(int(sys.version[0]) >= 3)


#############################################
#######   antgo parameters            #######
#############################################
DEFINE_string('project', '', 'set project name')
DEFINE_string('git', None, '')
DEFINE_string('branch', None, '')
DEFINE_string('commit', None, '')
DEFINE_string('image', '', '')                  # 镜像
DEFINE_int('cpu', 0, 'set cpu number')          # cpu 数
DEFINE_int('gpu', 0, 'set gpu number')          # gpu 数
DEFINE_int('memory', 0, 'set memory size (M)')  # 内存大小（单位M）
DEFINE_string('name', None, '')                 # 名字
DEFINE_string('type', None, '')                 # 类型
DEFINE_string("address", None, "")
DEFINE_indicator("auto", True, '')              # 是否项目自动优化
DEFINE_indicator("finetune", True, '')          # 是否启用finetune模式
DEFINE_string('id', None, '')
DEFINE_string("ip", "", "set ip")
DEFINE_int('port', 0, 'set port')
DEFINE_choices('stage', 'supervised', ['supervised', 'semi-supervised', 'distillation', 'activelearning', 'label'], '')
DEFINE_string('main', None, '')
DEFINE_indicator('data', True, '')
DEFINE_choices('mode', 'http', ['http', 'grpc', 'android/sdk', 'linux/sdk', 'windows/sdk', 'applet'], '')
DEFINE_string("image-repo", None, "image repo")
DEFINE_string("image-version", "latest", "image version")
DEFINE_string("user", None, "user name")
DEFINE_string("password", None, "user password")

############## submitter ###################
DEFINE_indicator('ssh', True, '')     # ssh 提交
DEFINE_indicator('k8s', True, '')     # k8s 提交
DEFINE_indicator('a', True, '')       # 配合ls命令使用，例如非活跃的实例

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
DEFINE_indicator("clear", True, "")   # 清理现场（用于远程提交时使用）

#############################################
DEFINE_nn_args()

action_level_1 = ['train', 'eval', 'export', 'config', 'server', 'activelearning', 'device', 'stop', 'ls', 'log', 'web', 'dataserver', 'deploy', 'package']
action_level_2 = ['add', 'del', 'create', 'register','update', 'show', 'get', 'tool', 'share', 'download', 'upload', 'submitter', 'dataset', 'metric', 'install']


def main():
  # 显示antgo logo
  main_logo()
  
  # 备份脚本参数
  sys_argv_cp = sys.argv
  if len(sys.argv) < 2:
    print('Use antgo help, show help list.')
    return

  # 解析参数
  action_name = sys.argv[1]
  sub_action_name = None
  if action_name in action_level_1:
    sys.argv = [sys.argv[0]] + sys.argv[2:]
  else:
    sub_action_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
  args = parse_args()

  if action_name == 'install':
    if sub_action_name not in ['eagleeye', 'opencv', 'eigen', 'grpc', 'ffmpeg']:
      print(f'sorry, {sub_action_name} not support.')
      return
    
    if sub_action_name == 'eagleeye':
      install_eagleeye()

    if sub_action_name == 'opencv':
      install_opencv()

    if sub_action_name == 'eigen':
      install_eigen()
    
    if sub_action_name == 'grpc':
      install_grpc()
    
    if sub_action_name == 'ffmpeg':
      install_ffmpeg()
    return

  if args.ip != '':
    args.ssh = True

  ######################################### 配置文件操作 ################################################
  # 检查配置文件是否存在
  if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
    # 使用指定配置文件更新
    if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
      os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

	  # 位置优先选择/data/, /HOME/
    config_data = {'FACTORY': '/data/.factory', 'USER_TOKEN': ''}
    if not os.path.exists('/data'):
      config_data['FACTORY'] = os.path.join(os.environ['HOME'], '.factory')

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

  # web服务
  if action_name == 'web':
    if args.port == 0:
      args.port = 8000

    if args.ip == "":
      args.ip = '0.0.0.0'
    os.system(f'uvicorn {args.main} --reload --port {args.port} --host {args.ip}')
    return

  # 镜像打包服务
  if action_name == 'package':
    if args.mode == 'http':
      # step 1: 构建Dockerfile
      logging.info('Generate Dockerfile')
      if args.version is None or args.version == '-' or args.version == '':
        args.version = 'master'
      dockerfile_data = {
        'version': args.version
      }
      env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
      dockerfile_template = env.get_template('script/Dockerfile')
      dockerfile_content = dockerfile_template.render(**dockerfile_data)
      with open('./Dockerfile', 'w') as fp:
        fp.write(dockerfile_content)

      logging.info('Generate Server Launch.sh')
      if args.main is None or args.main == '':
        logging.error('Must set main file.(--main=xxx)')
        return

      if args.port == 0:
        logging.error('Must set server port.(--port=8080)')
        return

      launch_tempate =  env.get_template('script/server-launch.sh')
      launch_data = {}      
      launch_data.update({
        'cmd': f'antgo web --main={args.main} --port={args.port}'
      })
      launch_content = launch_tempate.render(**launch_data)
      with open('./launch.sh', 'w') as fp:
        fp.write(launch_content)

      # step 2: 构建镜像
      if args.name is None or args.name == '':
        logging.error('Must set server image name. (--name=xxx)')
        return
      logging.info(f'Build docker image {args.name} (Server: {args.mode})')
      os.system(f'docker build -t {args.name} ./')

      # step 3: 发布镜像
      if args.image_repo is None or args.user is None or args.password is None:
        # logging.warn("No set image repo and user name, If need to deploy, must set --image-repo=xxxx --user=xxx --password=xxx.")
        logging.warn("No image_repo, only use local image file")
        image_time = time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time()))
        server_config_info = {
          'image_repo': '',
          'create_time': image_time,
          'update_time': image_time,
          'server_port': args.port,
          'name': args.name,
          'mode': args.mode
        }

        if os.path.exists('./server_config.json'):
          with open('./server_config.json', 'r') as fp:
            info = json.load(fp)
            server_config_info['create_time'] = info['create_time']

        # 更新服务配置
        with open('./server_config.json', 'w') as fp:
          json.dump(server_config_info, fp)
        return

      logging.info(f'Push image {args.name} to image repo {args.image_repo}:{args.image_version}')
      # 需要手动添加密码
      os.system(f'docker login --username={args.user} --password={args.password} {args.image_repo.split("/")[0]}')
      os.system(f'docker tag {args.name}:latest {args.image_repo}:{args.image_version}')
      os.system(f'docker push {args.image_repo}:{args.image_version}')

      image_time = time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time()))
      server_config_info = {
        'image_repo': f'{args.image_repo}:{args.image_version}',
        'create_time': image_time,
        'update_time': image_time,
        'server_port': args.port,
        'name': args.name,
        'mode': args.mode
      }

      if os.path.exists('./server_config.json'):
        with open('./server_config.json', 'r') as fp:
          info = json.load(fp)
          server_config_info['create_time'] = info['create_time']

      # 更新服务配置
      with open('./server_config.json', 'w') as fp:
        json.dump(server_config_info, fp)      
    elif args.mode == 'grpc':
      # step 1: 管线由C++代码构建，打包所有依赖
      import antgo.pipeline
      antgo.pipeline.pipeline_cplusplus_package(args.name)

      # step 2: 构建镜像，构建Dockerfile
      antgo.pipeline.pipeline_build_image(
        args.name, 
        version=args.version, 
        port=args.port,
        image_repo=args.image_repo,
        image_version=args.image_version,
        user=args.user,
        password=args.password,
      )
      return
    elif args.mode in['android/sdk', 'linux/sdk', 'windows/sdk']:
      # 管线由C++代码构建
      import antgo.pipeline
      antgo.pipeline.pipeline_cplusplus_package(args.name)
      return
    elif args.mode == 'applet':
      # 管线由C++代码构建
      return

    return

  # 镜像发布服务
  if action_name == 'deploy':
    # step 1: 检查是否存在指定镜像(需要给予镜像仓库)
    if not os.path.exists('./server_config.json'):
      logging.error('Need to antgo package ..., before.')
      return
    with open('./server_config.json', 'r') as fp:
      server_info = json.load(fp)

    if server_info['mode'] not in ['http', 'grpc']:
      logging.error('Only support mode = http, grpc')
      return

    try:
      assert('image_repo' in server_info)
      assert('server_port' in server_info)
      assert('create_time' in server_info)
      assert('update_time' in server_info)    
      assert('name' in server_info)  
    except:
      logging.error('Server info not complete. Need to antgo package xxx.')
      return

    print('Server Info.')
    print(server_info)

    # step 2: 远程启动服务
    if args.ssh:
      # 基于ssh远程部署
      if args.ip == '':
        logging.error('Must set remote ip (--ip=xxx).')
        return

      if args.port == 0:
        logging.error('Must set remote server port (--ip=xxx).')
        return

      ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{args.ip}-submit-config.yaml')
      if not os.path.exists(ssh_submit_config_file):
        logging.error(f'Dont exist ssh-{args.ip}-submit-config.yaml config, couldnt remote deploy')
        return
      with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
          ssh_config_info = yaml.safe_load(fp)

      if server_info['image_repo'] == '':
        # 将镜像本地打包，并传到目标机器
        os.system(f'docker save -o {server_info["name"]}.tar {server_info["name"]}')
        os.system(f'scp {server_info["name"]}.tar {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]}:~/')
        os.system(f'rm {server_info["name"]}.tar')

      # 生成服务部署脚本
      env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
      server_deploy_template = env.get_template('script/server-deploy.sh')
      server_deploy_data = {
        'user': args.user,
        'password': args.password,
        'image_registry': server_info['image_repo'].split('/')[0] if server_info['image_repo'] != '' else '\"\"',
        'image': server_info['image_repo'] if server_info['image_repo'] != '' else server_info['name'],
        'gpu_id': 0 if args.gpu_id == '' else args.gpu_id,
        'outer_port': args.port,
        'inner_port': server_info['server_port'],
        'name': server_info['name'],
        'workspace': '/workspace' if server_info['mode'] != 'grpc' else '/workspace/project/deploy/package/'
      }
      server_deploy_content = server_deploy_template.render(**server_deploy_data)

      print(server_deploy_content)

      with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, 'deploy.sh'), 'w') as fp:
          fp.write(server_deploy_content)
        deploy_cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} bash -s < {os.path.join(temp_dir, "deploy.sh")}'
        logging.info(deploy_cmd)
        os.system(deploy_cmd)
    elif args.k8s:
      logging.error('K8s deploy in comming.')
      return
    return

  # 查看运行设备（本地/远程）
  if action_name == 'device':
    if not (args.ssh or args.k8s):
      logging.info("Use default backend (SSH)")
      args.ssh = True
    
    tools.check_device_info(args)
    return

  # 操作执行任务（本地/远程）
  if action_name in ['stop', 'ls', 'log']:
    if not (args.ssh or args.k8s):
      logging.info("Use default backend (SSH)")
      args.ssh = True

    tools.operate_on_running_status(action_name, args)
    return

  if action_name == 'dataserver':
    # 数据生产服务
    if args.ssh:
      pass
    else:
      raise NotImplemented

    return

  ######################################### 支持扩展 ###############################################
  if args.extra and not os.path.exists('extra'):
    logging.info('download extra package')
    os.system('wget http://image.vibstring.com/extra.tar; tar -xf extra.tar; cd extra/manopth; python3 setup.py install')
  
  if args.ext_module != '':
    logging.info('import extent module')
    load_extmodule(args.ext_module)

  ##################################### 支持任务提交脚本配置  ###########################################
  if action_name == 'submitter':
    if not (args.ssh or args.k8s):
      logging.info("Use default backend (SSH)")
      args.ssh = True

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
        if not os.path.exists(os.path.join(os.environ['HOME'], '.ssh')):
          os.system(f'mkdir {os.path.join(os.environ["HOME"], ".ssh")}')

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
        os.system(f'cd {os.path.join(os.environ["HOME"], ".ssh")} && ssh -tt {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} < user_ssh_nopassword_config.sh')
      else:
        logging.error("Only support ssh remote task submitter (--ssh)")
        return

      print(f'Update & Activate submitter config {args.config}')
    elif sub_action_name == 'ls':
      # 输出已经配置的远程信息
      if args.ssh:
        for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
          if file_name.endswith('.yaml'):
            terms = file_name.split('-')
            if len(terms) == 4:
              pprint(f'{terms[1]}')
      else:
        logging.error("Only support ssh remote task submitter (--ssh)")
    else:
      logging.error("Only support submitter template/update/ls/activate")
      return

    return

  ##################################### 支持计算资源数据集操作 add/del/ls  ##############################
  if action_name == 'dataset':
    if sub_action_name is None or sub_action_name == '':
      logging.error(f'Only support {action_name} add/del/ls')
      return
    if args.src is None:
      logging.error(f'Need set --src=')
      return

    if not (args.ssh or args.k8s):
      logging.info("Use default backend (SSH)")
      args.ssh = True

    if sub_action_name == 'add':
      if args.ssh:
        if args.ip == '':
          for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
            register_ip = ''
            if file_name.endswith('.yaml') and file_name.startswith('ssh'):
                terms = file_name.split('-')
                if len(terms) == 4:
                    register_ip = terms[1]
            else:
                continue

            if register_ip == '':
                continue
          
            if args.ip == '':
              args.ip = register_ip
            else:
              args.ip = f'{args.ip},{register_ip}'

        deploy_ip_list = args.ip.split(',')
        print(f"Prepare deploy dataset to {deploy_ip_list}")

        for deploy_ip in deploy_ip_list:
          ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{deploy_ip}-submit-config.yaml')
          with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
              ssh_config_info = yaml.safe_load(fp)

          user_name = ssh_config_info['config']['username']
          remote_ip = ssh_config_info['config']['ip']

          if args.src.endswith('.tar'):
            os.system(f'scp {args.src} {user_name}@{remote_ip}:/data')
            os.system(f'ssh {user_name}@{remote_ip} "tar -xf /data/{os.path.basename(args.src)} -C /data/ && rm /data/{os.path.basename(args.src)}"')
          elif args.src.endswith('.zip'):
            os.system(f'scp {args.src} {user_name}@{remote_ip}:/data')
            os.system(f'ssh {user_name}@{remote_ip} "unzip -d /data/ /data/{os.path.basename(args.src)} && rm /data/{os.path.basename(args.src)}"')
          else:
            os.system(f'scp -r {args.src} {user_name}@{remote_ip}:/data')
          print(f'Finish dataset {os.path.basename(args.src)} deploy on IP {deploy_ip}.')
      else:
        logging.error("Now only support ssh remote control")
    elif sub_action_name == 'del':
      if '/' in args.src:
        logging.error('Only need set dataset name in --src, not a path')
        return

      if args.ssh:
        if args.ip == '':
          logging.error("Need set --ip=")
          return

        ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{args.ip}-submit-config.yaml')
        if not os.path.exists(ssh_submit_config_file):
            logging.error('No ssh submit config.')
            logging.error('Please run antgo submitter update --config= --ssh')
            return False

        with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
            config_content = yaml.safe_load(fp)

        user_name = config_content['config']['username']
        remote_ip = config_content['config']['ip']
        os.system(f'ssh {user_name}@{remote_ip} "rm -rf /data/{args.src}"')
        print(f'Finish dataset {args.src} delete On {args.ip}.')
      else:
        logging.error("Now only support ssh remote control")
    elif sub_action_name == 'ls':
      if args.ssh:
        if args.ip == '':
          for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
            register_ip = ''
            if file_name.endswith('.yaml') and file_name.startswith('ssh'):
                terms = file_name.split('-')
                if len(terms) == 4:
                    register_ip = terms[1]
            else:
                continue

            if register_ip == '':
                continue

            if args.ip == '':
              args.ip = register_ip
            else:
              args.ip = f'{args.ip},{register_ip}'

        for register_ip in args.ip.split(','):
          ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{register_ip}-submit-config.yaml')
          if not os.path.exists(ssh_submit_config_file):
              logging.error('No ssh submit config.')
              logging.error('Please run antgo submitter update --config= --ssh')
              return False

          with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
              config_content = yaml.safe_load(fp)

          user_name = config_content['config']['username']
          remote_ip = config_content['config']['ip']
          print(f"IP: {register_ip}")
          os.system(f'ssh {user_name}@{remote_ip} "ls -lh /data/"')
      else:
        logging.error("Now only support ssh remote control")
    else:
      logging.error("Only support dataset add/del/ls")
      return

    return

  #####################################   支持评估结果 ls  #############################################
  if action_name == 'metric':
    from antgo.tools.display_funcs import table_show
    if sub_action_name == 'ls':
      if os.path.exists('./.project.json'):
        with open('./.project.json', 'r') as fp:
          project_info = json.load(fp)

      logging.info(f'list model metric for {args.config}')
      show_exp_metric_list = []
      for exp_name, exp_info_list in project_info['exp'].items():
        for exp_info in exp_info_list:
          if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
            if 'metric' in exp_info:
              for metric_info in exp_info['metric']:
                show_exp_metric_list.append(
                  {
                    'name': exp_name,
                    'create_time': exp_info['create_time'],
                    'eval_time': metric_info['time'],
                    'root': exp_info['root'],
                    'checkpoint': metric_info['checkpoint'].split('/')[-1],
                    'metric': str(metric_info['metric'])
                  }
                )

      if len(show_exp_metric_list) == 0:
        logging.warn("Dont have exp metric")
        return

      table_show(show_exp_metric_list, ['name', 'create_time', 'eval_time', 'root', 'checkpoint', 'metric'])
    else:
      logging.error("Only support metric ls")

    return

  ######################################### ROOT 设置 #################################################
  if args.root is None or args.root == '':
    print('Using default root address ali:///exp')
    args.root = "ali:///exp"

  ######################################### 检查token #################################################
  # 解析配置文件
  config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
  config.AntConfig.parse_xml(config_xml)

  if not os.path.exists(config.AntConfig.factory):
    os.makedirs(config.AntConfig.factory)
  if not os.path.exists(config.AntConfig.data_factory):
    os.makedirs(config.AntConfig.data_factory)
  if not os.path.exists(config.AntConfig.task_factory):
    os.makedirs(config.AntConfig.task_factory)

  # 检查token是否存在，否则重新生成
  token = None
  if os.path.exists('./.token'):
      with open('./.token', 'r') as fp:
          token = fp.readline()
  else:
    token = getattr(config.AntConfig, 'server_user_token', '')

  if token is None or token == '':
    logging.info("generate experiment token")
    token = mlogger.create_token()
    config_data = {
      'FACTORY': getattr(config.AntConfig, 'factory'), 
      'USER_TOKEN': token
    }

    env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
    config_template = env.get_template('config.xml')
    config_content = config_template.render(**config_data)

    with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'), 'w') as fp:
      fp.write(config_content)
    logging.warn(f'update config file (token: {config_data["USER_TOKEN"]}, factory: {config_data["FACTORY"]}).')

  with open('./.token', 'w') as fp:
    fp.write(token)

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
    if args.root.endswith('/'):
      args.root = args.root[:-1]

    if not os.path.exists('./.project.json'):
      shutil.copyfile(os.path.join(os.path.dirname(__file__), 'resource', 'templates', 'project.json'), './.project.json')

    if args.ssh or args.k8s:
      if args.root.startswith('ali:'):
        # 尝试进行认证，从而保证当前路径下生成认证信息
        ali = Aligo()
        shutil.copy(os.path.join(Path.home().joinpath('.aligo'), 'aligo.json'),'./')

      # 基于实验名称和时间戳修改实验root
      now_time = time.time()

      with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)

      if action_name in ['train', 'activelearning']:
        # train, activelearning
        # 项目基本信息
        project_info['image'] = args.image      # 镜像名称
        if args.exp not in project_info['exp']:
          project_info['exp'][args.exp] = []

        # root 地址
        args.root = f'{args.root}/{args.exp}/'+time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time))

        # 配置新增实验信息
        project_info['exp'][args.exp].append({
          'id': '',
          'ip': '',
          'create_time': time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time)),
          'config': args.config,
          'root': args.root
        })
        with open('./.project.json', 'w') as fp:
          json.dump(project_info,fp)
      else:
        # eval, export
        # 匹配实验记录（exp, config）
        # (1) root 匹配
        # (2) 默认匹配最新实验
        if args.exp not in project_info['exp']:
          logging.error(f'{args.exp} not in project')
          return

        if len(project_info['exp'][args.exp]) == 0:
          logging.error(f'{args.exp} dont have any record in project')
          return

        found_exp_info = None
        for exp_info in project_info['exp'][args.exp]:
          if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
            continue

          if exp_info['root'] == args.root:
            found_exp_info = exp_info
            break

        if found_exp_info is None:
          args.root = project_info['exp'][args.exp][-1]['root']

      filter_sys_argv_cp = []
      for t in sys_argv_cp:
        if t.startswith('--project'):
          continue
        if t.startswith('--root'):
          continue
        if t.startswith('--gpu-id'):
          continue
        if t.startswith('--ip'):
          continue
        if t.startswith('--node-rank'):
          continue
        if t.startswith('--nodes'):
          continue
        if t.startswith('--version'):
          continue
        filter_sys_argv_cp.append(t)

      sys_argv_cp = filter_sys_argv_cp
      sys_argv_cp.append(f'--root={args.root}')
      sys_argv_cmd = ' '.join(sys_argv_cp[1:])

      # 直接进行任务提交
      # step 1.1: 检查提交脚本配置
      if args.ssh:
        # ssh提交
        sys_argv_cmd = sys_argv_cmd.replace('--ssh', '')
        sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
        sys_argv_cmd = f'antgo {sys_argv_cmd}'

        ssh_submit_process_func(args.project, time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time)), sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory, ip=args.ip, exp=args.exp, check_data=args.data, env=args.version)
      else:
        # 自定义脚本提交
        sys_argv_cmd = sys_argv_cmd.replace('  ', ' ')
        sys_argv_cmd = f'antgo {sys_argv_cmd}'          
        custom_submit_process_func(args.project, time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time)), sys_argv_cmd, 0 if args.gpu_id == '' else len(args.gpu_id.split(',')), args.cpu, args.memory, ip=args.ip, exp=args.exp, check_data=args.data)

      # 清理临时存储信息
      if os.path.exists('./aligo.json'):
        os.remove('./aligo.json')
      return

    if args.root.startswith('ali:'):
      # 尝试使用缓存的信息，进行认证
      if os.path.exists("./aligo.json"):
        if not os.path.exists(Path.home().joinpath('.aligo')):
          os.makedirs(Path.home().joinpath('.aligo'))
        shutil.copy('./aligo.json', Path.home().joinpath('.aligo'))
      ali = Aligo()

    now_time = time.time()
    # 实验依赖资源（允许远程存储）
    if args.checkpoint is None:
      args.checkpoint = ''
    if args.resume_from is None:
      args.resume_from = ''

    # 下载依赖checkpoint
    if args.checkpoint != '':
      # checkpoint路径格式
      # 1: local path                 本地目录
      # 2: ali://                     直接从阿里云盘下载
      # 3: experiment/checkpoint      日志平台（推荐）      
      print(args.checkpoint)
      if args.checkpoint.startswith('ali://'):
        # 阿里云盘路径
        if not os.path.exists('./checkpoint'):
          os.makedirs('./checkpoint')

        with FileLock('download.lock'):
          checkpoint_name = args.checkpoint.split('/')[-1]
          if not os.path.exists(os.path.join('./checkpoint', checkpoint_name)):
            logging.info('downling checkpoint...')
            logging.info(args.checkpoint)              
            file_client_get(args.checkpoint, './checkpoint')
          args.checkpoint = os.path.join('./checkpoint', checkpoint_name)

      # checkpoint（可能具体路径存储在日志平台，需要实际运行时获取具体路径并下载）
      logging.info(f'use checkpoint {args.checkpoint}')

    if args.resume_from != '':
      if not os.path.exists(args.resume_from):
        # 非本地有效路径
        if not os.path.exists('./checkpoint'):
          os.makedirs('./checkpoint')
        with FileLock('download.lock'):
          if '/' in args.resume_from:
            # 如果传入的是路径，则尝试直接下载
            checkpoint_name = args.resume_from.split('/')[-1]
            if not os.path.exists(os.path.join('./checkpoint', checkpoint_name)):
              logging.info('downling checkpoint...')
              logging.info(args.resume_from)              
              file_client_get(args.resume_from, './checkpoint')
            args.resume_from = os.path.join('./checkpoint', checkpoint_name)
          else:
            # 尝试从实验存储目录中，加载
            if os.path.exists('./.project.json'):
              with open('./.project.json', 'r') as fp:
                project_info = json.load(fp)

            checkpoint_name = args.resume_from
            if args.exp in project_info['exp']:
              found_exp_related_info_list = []
              for exp_info in project_info['exp'][args.exp]:
                if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
                  found_exp_related_info_list.append(
                    exp_info
                  )

              found_exp_info = None
              if len(found_exp_related_info_list) > 0:
                found_exp_info = found_exp_related_info_list[-1]

              if found_exp_info is not None:
                found_exp_root = found_exp_info['root']
                logging.info('downling checkpoint...')
                logging.info(f'{found_exp_root}/output/checkpoint/{checkpoint_name}')
                file_client_get(f'{found_exp_root}/output/checkpoint/{checkpoint_name}', './checkpoint')
                args.resume_from = os.path.join('./checkpoint', checkpoint_name)

      logging.info(f'use resume_from {args.resume_from}')
      if not os.path.exists(args.resume_from):
        logging.error(f'Dont exist {args.resume_from}, exit.')
        return

    # 执行任务
    auto_exp_name = f'{args.exp}.{args.id}' if args.id is not None else args.exp
    script_folder = os.path.join(os.path.dirname(__file__), 'script')
    if action_name == 'train':
      if args.exp not in args.root:
        # root需要将exp加入点到root中
        args.root = f'{args.root}/{args.exp}/'+time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time))

      # 为训练实验，创建存储root
      file_client_mkdir(args.root)

      # 配置工程文件
      with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)
      if args.exp not in project_info['exp']:
        project_info['exp'][args.exp] = []
      
      project_info['exp'][args.exp].append({
        'id': f'{os.getpid()}',
        'ip': '',
        'create_time': time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time)),
        'config': args.config,
        'root': args.root
      })

      with open('./.project.json', 'w') as fp:
        json.dump(project_info,fp)

      # 训练过程
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
        if args.find_unused_parameters:
          command_str += f' --find-unused-parameters'

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
        if args.find_unused_parameters:
          command_str += f' --find-unused-parameters'

        os.system(command_str)
      else:
        # multi gpu run
        # (1)安装;(2)数据准备;(3)运行
        gpu_num = len(args.gpu_id.split(','))
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint} --resume-from={args.resume_from}; bash launch.sh {args.exp}/main.py {gpu_num} {args.nodes} {args.node_rank} {args.master_addr} --exp={auto_exp_name} --process=train --root={args.root} --extra-config={args.extra_config} --config={args.config}'
        if args.no_validate:
          command_str += ' --no-validate'
        if args.resume_from is not None:
          command_str += f' --resume-from={args.resume_from}'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        if args.max_epochs > 0:
          command_str += f' --max-epochs={args.max_epochs}'
        if args.find_unused_parameters:
          command_str += f' --find-unused-parameters'
        if args.diff_seed:
          command_str += f' --diff-seed'
        os.system(command_str)
    elif action_name == 'activelearning':
      if args.exp not in args.root:
        # root需要将exp加入点到root中
        args.root = f'{args.root}/{args.exp}/'+time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(now_time))

      # 为主动学习实验，创建存储root
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
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; bash launch.sh {args.exp}/main.py {gpu_num} {args.nodes} {args.node_rank} {args.master_addr} --exp={auto_exp_name}  --process=activelearning --root={args.root} --extra-config={args.extra_config} --config={args.config}'
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
      if args.checkpoint is None or args.checkpoint == '':
        logging.error('Must set --checkpoint=')
        return

      # 删除存在的历史评估缓存结果
      if os.path.exists('./evalresult.json'):
        os.remove('./evalresult.json')

      # 获得实验root
      if args.exp not in args.root:
        with open('./.project.json', 'r') as fp:
          project_info = json.load(fp)
        if args.exp not in project_info['exp']:
          logging.error(f'{args.exp} not in project')
          return

        if len(project_info['exp'][args.exp]) == 0:
          logging.error(f'{args.exp} dont have any record in project')
          return

        found_exp_info = None
        for exp_info in project_info['exp'][args.exp]:
          if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
            continue

          if exp_info['root'] == args.root:
            found_exp_info = exp_info
            break

        if found_exp_info is None:
          args.root = project_info['exp'][args.exp][-1]['root']

      # (1)安装;(2)数据准备;(3)运行
      if args.gpu_id == '' or int(args.gpu_id.split(',')[0]) == -1:
        # cpu run
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={-1} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config} --json=evalresult.json'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      elif len(args.gpu_id.split(',')) == 1:
        # single gpu run
        gpu_id = args.gpu_id.split(',')[0]
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; python3 {args.exp}/main.py --exp={auto_exp_name} --gpu-id={gpu_id} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config} --json=evalresult.json'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)
      else:
        # multi gpu run
        gpu_num = len(args.gpu_id.split(','))
        command_str = f'bash install.sh; python3 {script_folder}/data_prepare.py --exp={args.exp} --extra-config={args.extra_config} --config={args.config} --checkpoint={args.checkpoint}; bash launch.sh {args.exp}/main.py {gpu_num} {args.nodes} {args.node_rank} {args.master_addr} --exp={auto_exp_name} --process=test --root={args.root} --extra-config={args.extra_config} --config={args.config} --json=evalresult.json'
        if args.checkpoint is not None:
          command_str += f' --checkpoint={args.checkpoint}'
        os.system(command_str)

      # 记录评估结果
      with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)

      # 评估结果记录到项目信息中
      if args.exp in project_info['exp']:
          for exp_info in project_info['exp'][args.exp]:
            if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
              if 'metric' not in exp_info:
                exp_info['metric'] = []

              # {'checkpoint': '', 'metric': {}, 'time': ''}
              if not os.path.exists('./evalresult.json'):
                logging.error("Not found eval result file.")
              else:
                with open("./evalresult.json", 'r') as fp:
                  metric_info = json.load(fp)

              exp_info['metric'].append(
                {
                  'checkpoint': args.checkpoint,
                  'time': time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time())),
                  'metric': metric_info
                }
              )
              break

      with open('./.project.json', 'w') as fp:
        json.dump(project_info, fp)

      # 删除评估缓存结果
      if os.path.exists('./evalresult.json'):
        os.remove('./evalresult.json')
    elif action_name == 'export':
      if args.checkpoint is None or args.checkpoint == '':
        logging.error('Must set --checkpoint=')
        return

      # 获得实验root
      if args.exp not in args.root:
        with open('./.project.json', 'r') as fp:
          project_info = json.load(fp)
        if args.exp not in project_info['exp']:
          logging.error(f'{args.exp} not in project')
          return

        if len(project_info['exp'][args.exp]) == 0:
          logging.error(f'{args.exp} dont have any record in project')
          return

        found_exp_info = None
        for exp_info in project_info['exp'][args.exp]:
          if exp_info['config'].split('/')[-1] == args.config.split('/')[-1]:
            continue

          if exp_info['root'] == args.root:
            found_exp_info = exp_info
            break

        if found_exp_info is None:
          args.root = project_info['exp'][args.exp][-1]['root']

      os.system(f'bash install.sh; python3 {args.exp}/main.py --exp={auto_exp_name} --checkpoint={args.checkpoint} --process=export --root={args.root} --config={args.config} --work-dir={args.work_dir}')
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
        if args.num == 0:
          args.num = 1000
          print(f'Default download sample numbe 1000 (--num=...)')
        tool_func(args.tgt, args.tags, src_path=args.src, target_num=args.num)
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
