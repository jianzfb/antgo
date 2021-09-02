# encoding=utf-8
# @Time    : 17-3-3
# @File    : main.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from typing import NamedTuple
sys.path.append('/workspace/workspace/portrait_code/tool/antgo')
sys.path.append('/workspace/workspace/portrait_code/tool/antgo/antgo')
from antgo.ant.generate import *
from antgo.ant.demo import *
from antgo.ant.train import *
from antgo.ant.challenge import *
from antgo.ant.browser import *
from antgo.ant.batch import *
from antgo.ant.activelearning import *
from antgo.ant import activelearning_api
from antgo.ant.utils import *
from antgo.sandbox.sandbox import *
from antgo.utils.utils import *
from antgo.utils.dht import *
from antgo import version
import traceback
from jinja2 import Environment, FileSystemLoader

import multiprocessing
import subprocess

if sys.version > '3':
    PY3 = True
else:
    PY3 = False


def _check_environment():
  is_in_mltalker = True if os.environ.get('ANT_ENVIRONMENT', '') != '' else False
  return is_in_mltalker


_ant_support_commands = ["train",
                         "challenge",
                         "activelearning",
                         "dataset",
                         "predict",
                         "demo",
                         "browser",
                         "startproject",
                         "config"]

#############################################
#######   antgo parameters            #######
#############################################
flags.DEFINE_string('main_file', None, 'main file')
flags.DEFINE_string('main_param', None, 'model parameters')
flags.DEFINE_string('main_folder', None, 'resource folder')
flags.DEFINE_string('version', None, 'minist antgo version')
flags.DEFINE_string('task', None, 'task file')
flags.DEFINE_string('task_t', '', 'task type')
flags.DEFINE_string('task_ep', '', 'holdout/repeated-holdout/bootstrap/kfold')
flags.DEFINE_string('task_em', '', 'define based on task type')
flags.DEFINE_integer('task_badcase_num', 100, 'badcase analysis (num)')
flags.DEFINE_integer('task_badcase_category', 1, 'badcase analysis (category)')
flags.DEFINE_string('dataset', None, 'dataset name')
flags.DEFINE_string('file', '', '')
flags.DEFINE_string('signature', '123', 'signature')
flags.DEFINE_string('devices', '', 'devices')
flags.DEFINE_string('servers', '', '')
flags.DEFINE_string('dump', None, 'dump dir')
flags.DEFINE_string('token', None, 'token')
flags.DEFINE_string('proxy', None, 'proxy')
flags.DEFINE_string('name', None, 'name')
flags.DEFINE_string('author', None, 'author')
flags.DEFINE_string('net', None, 'GENERAL')
flags.DEFINE_string('max_time', '100d', 'max running time')
flags.DEFINE_string('from_experiment', None, 'load model from experiment')
flags.DEFINE_string('restore_experiment', None, 'restore experiment')
flags.DEFINE_string('factory', None, '')
flags.DEFINE_string('config', None, 'config file')
flags.DEFINE_string('benchmark', None, 'benchmark experiments')
flags.DEFINE_string('host_ip', '127.0.0.1', 'host ip address')
flags.DEFINE_integer('host_port', -1, 'port')
flags.DEFINE_string('html_template', None, 'html template')
flags.DEFINE_indicator('worker', '')
flags.DEFINE_indicator('master', '')
flags.DEFINE_indicator('unlabel', '')
flags.DEFINE_indicator('skip_training', '')
flags.DEFINE_indicator('research', 'research mode or not')
flags.DEFINE_string('running_platform', 'local', 'local/cloud')
flags.DEFINE_string('param', '', '')  # k:v;k:v


FLAGS = flags.AntFLAGS
Config = config.AntConfig


def main():
  # 1.step antgo logo
  main_logo()

  # 3.step parse antgo running params
  if len(sys.argv) >= 2:
    if sys.argv[1].startswith('--') or sys.argv[1].startswith('-'):
      flags.cli_param_flags(sys.argv[1:])
    else:
      flags.cli_param_flags(sys.argv[2:])

  # token
  token = FLAGS.token()
  if not PY3 and token is not None:
    token = unicode(token)

  if sys.argv[1] == 'config':
    if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
      os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

    if FLAGS.config() is not None:
      # 使用指定配置文件更新
      shutil.copy(FLAGS.config(), os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'))
      logger.info('Success update config file.')
    else:
      # 在当前目录生成默认配置文件
      # --dataset, --token, 自动配置
      config_data = {'DATASET_FACTORY': '', 'USER_TOKEN': ''}
      if FLAGS.dataset() is not None:
        if not os.path.exists(FLAGS.dataset()):
          logger.error('Dataset factory path dont exist.')
          return
        config_data['DATASET_FACTORY'] = FLAGS.dataset()

      if FLAGS.token() is not None:
         config_data['USER_TOKEN'] = FLAGS.token()

      env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-1])))
      config_template = env.get_template('config.xml')
      config_content = config_template.render(**config_data)

      if config_data['DATASET_FACTORY'] != '':
        with open(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'),'w') as fp:
          fp.write(config_content)
        
        logger.info('Finish antgo global config.')
      else:
        with open('./config.xml','w') as fp:
          fp.write(config_content)

        logger.info('Please fill ./config.xml, then call (antgo config --config=./config.xml) to finish global config.')
    return

  # 检查配置文件是否存在
  if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
    logger.error('Missing config file, please run antgo config.')
    return

  # 解析配置文件
  config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
  Config.parse_xml(config_xml)

  # 检查最低版本与当前版本的兼容性
  if FLAGS.running_platform == 'local' and FLAGS.version() is not None:
    a, b, c = FLAGS.version().split('.')
    sys_a, sys_b, sys_c = version.split('.')
    if int(sys_a) < int(a) or int(sys_b) < int(b) or int(sys_c) < int(c):
      logger.error('Antgo version dont satisfy task minimum request (%s).'%FLAGS.version())
      return

  # 4.step check factory
  factory = getattr(Config, 'factory', None)
  if factory is None or \
      factory == '' or not os.path.exists(Config.factory):
    logger.error('Factory folder is missing, please run antgo config.')
    return

  if not os.path.exists(Config.data_factory):
    os.makedirs(Config.data_factory)
  if not os.path.exists(Config.task_factory):
    os.makedirs(Config.task_factory)

  task_factory = Config.task_factory
  data_factory = Config.data_factory

  # 5.step parse antgo execute command
  # 5.1.step other command (train, challenge, compose, deploy, server)
  ant_cmd = sys.argv[1]
  ant_cmd = ant_cmd.split('/')[0]
  ant_cmd_api = '' if sys.argv[1] == ant_cmd else sys.argv[1].split('/')[-1]
  if ant_cmd not in _ant_support_commands:
    logger.error('Antgo cli support( %s )command.'%",".join(_ant_support_commands))
    return
  
  if ant_cmd_api != '':
    # api 
    if ant_cmd == 'activelearning':
      func = getattr(activelearning_api, 'activelearning_api_'+ant_cmd_api, None)
      if func is None:
        logger.error('%s/%s dont support.'%(ant_cmd, ant_cmd_api))
        return
      
      func()
    return

  if ant_cmd == 'startproject':
    project_name = FLAGS.name()
    if project_name == "":
      project_name = 'AntgoProject'

    if os.path.exists(os.path.join(os.curdir, project_name)):
      while True:
        project_i = 0
        if not os.path.exists(os.path.join(os.curdir, '%s-%d'%(project_name, project_i))):
          project_name = '%s-%d'%(project_name, project_i)
          break

        project_i += 1

    os.makedirs(os.path.join(os.curdir, project_name))

    # generate main file and main param templates
    template_file_folder = os.path.join(os.path.dirname(__file__), 'resource', 'templates')
    file_loader = FileSystemLoader(template_file_folder)
    env = Environment(loader=file_loader)
    template = env.get_template('task_main_file.template')
    output = template.render(ModelTime=datetime.fromtimestamp(timestamp()).strftime('%Y-%m-%d'),
                             ModelName=FLAGS.name(),
                             ModelAuthor=FLAGS.author() if FLAGS.author() is not None else 'xxx',
                             tensorflow=False,
                             GAN=False)

    with open(os.path.join(os.curdir, project_name, '%s_main.py' % FLAGS.name()),'w') as fp:
      fp.write(output)

    template = env.get_template('task_main_param.template')
    output = template.render(ModelTime=datetime.fromtimestamp(timestamp()).strftime('%Y-%m-%d'),
                             ModelName=FLAGS.name(),
                             ModelAuthor=FLAGS.author() if FLAGS.author() is not None else 'xxx')
    with open(os.path.join(os.curdir, project_name, '%s_param.yaml' % FLAGS.name()),'w') as fp:
      fp.write(output)

    template = env.get_template('task.template')
    output = template.render(ModelName=FLAGS.name())
    with open(os.path.join(os.curdir, project_name, '%s_task.xml' % FLAGS.name()),'w') as fp:
      fp.write(output)

    template = env.get_template('task_shell.template')
    output = ''
    if FLAGS.token() is not None:
      output = template.render(token=FLAGS.token(),
                               main_file='%s_main.py' % FLAGS.name(),
                               main_param='%s_param.yaml' % FLAGS.name())
    else:
      output = template.render(task='%s_task.xml' % FLAGS.name(),
                               main_file='%s_main.py' % FLAGS.name(),
                               main_param='%s_param.yaml' % FLAGS.name())

    with open(os.path.join(os.curdir, project_name, 'run.sh'), 'w') as fp:
      fp.write(output)

    return

  # 7.step check related params
  # 7.1 must set name
  name = FLAGS.name()
  if name is None:
    logger.error('Must set experiemnt name.')
    return

  # 7.2 check main folder (all related model code, includes main_file and main_param)
  main_folder = FLAGS.main_folder()
  if main_folder is None:
    main_folder = os.path.abspath(os.curdir)

  if os.path.exists(os.path.join(main_folder, '%s.tar.gz'%name)):
    untar_shell = 'openssl enc -d -aes256 -in %s.tar.gz -k %s | tar xz -C %s' % (name, FLAGS.signature(), '.')
    subprocess.call(untar_shell, shell=True, cwd=main_folder)
    os.remove(os.path.join(main_folder, '%s.tar.gz' % name))

    main_folder = os.path.join(main_folder, name)
    os.chdir(main_folder)

  # 7.3 check dump dir (all running data is stored here)
  dump_dir = FLAGS.dump()
  if dump_dir is None:
    dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

  # 7.4 check main file
  main_file = FLAGS.main_file()
  if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
    if not (FLAGS.worker() or FLAGS.master()):
      logger.error('Main executing file dont exist.')
      return

  # 8 step ant running
  # 8.1 step what is task
  task = FLAGS.task()
  if task is not None:
    if os.path.exists(os.path.join(task_factory, task)):
      task = os.path.join(task_factory, task)
    else:
      task = os.path.join(main_folder, task)

  dataset = FLAGS.dataset()
  if task is None and dataset is not None:
    # build default task
    with open(os.path.join(task_factory, '%s.xml'%name), 'w') as fp:
      task_content = '<task><task_name>%s</task_name><task_type>%s</task_type><task_badcase><badcase_num>%d</badcase_num><badcase_category>%d</badcase_category></task_badcase>' \
                     '<input>' \
                     '<source_data><data_set_name>%s</data_set_name>' \
                     '</source_data>' \
                     '<estimation_procedure><type>%s</type></estimation_procedure>' \
                     '<evaluation_measures><evaluation_measure>%s</evaluation_measure></evaluation_measures>' \
                     '</input>' \
                     '</task>'%(name, FLAGS.task_t(), FLAGS.task_badcase_num(), FLAGS.task_badcase_category(), dataset, FLAGS.task_ep(), FLAGS.task_em())
      fp.write(task_content)
    task = os.path.join(task_factory, '%s.xml'%name)

  # 8.2 step load ant context
  ant_context = None
  if main_file is not None and main_file != '':
    try:
      ant_context = main_context(main_file, main_folder)
    except Exception as e:
      traceback.print_exc()
      print('Fail to load main_file %s.'%main_file)
      return
  else:
    ant_context = Context()
  
  # 8.3 step load model config
  main_param = FLAGS.main_param()
  if main_param is not None:
    main_config_path = os.path.join(main_folder, main_param)
    try:
      with open(main_config_path, 'r') as fp:
        # 用户配置参数
        params = yaml.load(fp, Loader=yaml.FullLoader)

        # 系统全局参数
        params.update({
          'system': {'research': FLAGS.research(), 
                    'skip_training': FLAGS.skip_training(),
                    'ip': FLAGS.host_ip(),
                    'port': FLAGS.host_port(),
                    'devices': FLAGS.devices(),
                    'running_platform': FLAGS.running_platform()},
        })
        ant_context.params = params
    except Exception as e:
      traceback.print_exc()
      print('Fail to load main_param %s.'%main_param)
      return

  ant_context.name = name
  ant_context.data_factory = data_factory
  ant_context.task_factory = task_factory

  # 8.4 step load experiment
  if FLAGS.from_experiment() is not None and \
          FLAGS.running_platform() == 'local' and \
          ant_cmd not in ['release']:
    experiment_path = ''
    if os.path.isdir(FLAGS.from_experiment()):
      # 绝对路径指定的实验
      experiment_path = FLAGS.from_experiment()
    else:
      experiment_path = os.path.join(dump_dir, FLAGS.from_experiment())

    if not os.path.exists(experiment_path):
      logger.error('Couldnt find experiment %s.'%FLAGS.from_experiment())
      return

    # 设置实验目录
    ant_context.from_experiment = experiment_path

  # time stamp
  time_stamp = timestamp()
  if ant_cmd == "train":
    with running_sandbox(sandbox_time=FLAGS.max_time(),
                         sandbox_dump_dir=main_folder,
                         sandbox_experiment=None,
                         sandbox_user_token=token,
                         sandbox_user_proxy=FLAGS.proxy(),
                         sandbox_user_signature=FLAGS.signature()):
      running_process = AntTrain(ant_context,
                                 name,
                                 data_factory,
                                 dump_dir,
                                 token,
                                 task,
                                 main_file=main_file,
                                 main_folder=main_folder,
                                 main_param=main_param,
                                 time_stamp=time_stamp,
                                 skip_training=FLAGS.skip_training(),
                                 running_platform=FLAGS.running_platform(),
                                 proxy=FLAGS.proxy(),
                                 signature=FLAGS.signature(),
                                 devices=FLAGS.devices())
      running_process.start()
  elif ant_cmd == 'challenge':
    with running_sandbox(sandbox_dump_dir=main_folder,
                         sandbox_experiment=None,
                         sandbox_user_token=token,
                         sandbox_user_proxy=FLAGS.proxy(),
                         sandbox_user_signature=FLAGS.signature()):
      running_process = AntChallenge(ant_context,
                                     name,
                                     data_factory,
                                     dump_dir,
                                     token,
                                     task,
                                     FLAGS.benchmark(),
                                     main_file=main_file,
                                     main_folder=main_folder,
                                     main_param=main_param,
                                     time_stamp=time_stamp,
                                     running_platform=FLAGS.running_platform(),
                                     devices=FLAGS.devices())
      running_process.start()
  elif ant_cmd == "demo":
    running_process = AntDemo(ant_context,
                              name,
                              dump_dir,
                              token,
                              task,
                              html_template=FLAGS.html_template(),
                              ip=FLAGS.host_ip(),
                              port=FLAGS.host_port(),
                              time_stamp=time_stamp,
                              devices=FLAGS.devices())

    running_process.start()
  elif ant_cmd == "predict":
    running_process = AntBatch(ant_context,
                               name,
                               FLAGS.host_ip(),
                               FLAGS.host_port(),
                               token,
                               data_factory,
                               dump_dir,
                               task,
                               unlabel=FLAGS.unlabel(),
                               devices=FLAGS.devices(),
                               restore_experiment=FLAGS.restore_experiment())
    running_process.start()
  elif ant_cmd == "activelearning":
    running_process = AntActiveLearning(ant_context,
                                        name,
                                        data_factory,
                                        dump_dir,
                                        token,
                                        task,
                                        main_file=main_file,
                                        main_folder=main_folder,
                                        main_param=main_param,
                                        time_stamp=time_stamp)
    running_process.start()
  elif ant_cmd == 'dataset':
    running_process = AntGenerate(ant_context,
                                  name,
                                  data_factory,
                                  dump_dir,
                                  token,
                                  dataset)
    running_process.start()
  elif ant_cmd == 'browser':
    running_process = AntBrowser(ant_context,
                                 name,
                                 token,
                                 FLAGS.host_ip(),
                                 FLAGS.host_port(),
                                 data_factory,
                                 dataset,
                                 dump_dir)
    running_process.start()

  # 9.step clear context
  ant_context.wait_until_clear()

if __name__ == '__main__':
  main()
