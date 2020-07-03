# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append("/Users/zhangjian52/Downloads/workspace/code/antvis/")
from antgo.ant.shell import *
from antgo.ant.generate import *
from antgo.ant.demo import *
from antgo.ant.browser import *
from antgo.ant.batch import *
from antgo.ant.activelearning import *
from antgo.ant.utils import *
from antgo.sandbox.sandbox import *
from antgo.utils.utils import *
from antgo.utils.dht import *
from antgo import version
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
                         "batch",
                         "demo",
                         "browser",
                         "startproject",
                         "tools/tffrozen",
                         "tools/tfrecords",
                         "tools/tfgraph"]

#############################################
#######   antgo parameters            #######
#############################################
flags.DEFINE_string('main_file', None, 'main file')
flags.DEFINE_string('main_param', None, 'model parameters')
flags.DEFINE_string('main_folder', None, 'resource folder')
flags.DEFINE_string('version', None, 'minist antgo version')
flags.DEFINE_string('task', None, 'task file')
flags.DEFINE_string('dataset', None, 'dataset')
flags.DEFINE_string('signature', '123', 'signature')
flags.DEFINE_string('devices', '', 'devices')
flags.DEFINE_string('servers', '', '')
flags.DEFINE_string('dump', None, 'dump dir')
flags.DEFINE_string('token', None, 'token')
flags.DEFINE_string('proxy', None, 'proxy')
flags.DEFINE_string('name', None, 'name')
flags.DEFINE_string('author', None, 'author')
flags.DEFINE_string('framework', None, 'tensorflow')
flags.DEFINE_string('net', None, 'GENERAL')
flags.DEFINE_string('max_time', '100d', 'max running time')
flags.DEFINE_string('from_experiment', None, 'load model from experiment')
flags.DEFINE_string('factory', None, '')
flags.DEFINE_string('config', None, 'config file')
flags.DEFINE_string('benchmark', None, 'benchmark experiments')
flags.DEFINE_string('port', 10000, 'port')
flags.DEFINE_string('html_template', None, 'html template')
flags.DEFINE_string('option', '', '')
flags.DEFINE_indicator('worker', '')
flags.DEFINE_indicator('master', '')
flags.DEFINE_indicator('unlabel', '')
flags.DEFINE_indicator('zoo', '')
flags.DEFINE_indicator('skip_training', '')
flags.DEFINE_string('running_platform', 'local', 'local/cloud')
flags.DEFINE_string('order_id', '', '')
flags.DEFINE_string('order_ip', '', '')
flags.DEFINE_integer('order_rpc_port', 0, '')
flags.DEFINE_integer('order_ssh_port', 0, '')
flags.DEFINE_float('max_fee', 0.0, '')
#############################################
########  tools - tffrozen            #######
#############################################
flags.DEFINE_string('tffrozen_input_nodes', '', 'input node names in graph')
flags.DEFINE_string('tffrozen_output_nodes', '', 'output node names in graph')
flags.DEFINE_string('tfgraph_path', '', 'pb file path')

FLAGS = flags.AntFLAGS
Config = config.AntConfig


def main():
  # 1.step antgo logo
  main_logo()

  # 2.step check antgo support command
  if len(sys.argv) >= 2 and \
          ((not sys.argv[1].startswith('-')) and sys.argv[1] not in _ant_support_commands):
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    sys.exit(-1)

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

  # load antgo global config
  if FLAGS.config() is not None:
    # 1.step parse config.xml
    Config.parse_xml(FLAGS.config())
    # 2.step try copy to system
    try:
      # shutil.copy(FLAGS.config(), os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-1]), 'config.xml'))
      # copy to ~/.config/
      if not os.path.exists(os.path.join(os.environ['HOME'],'.config','antgo')):
        os.makedirs(os.path.join(os.environ['HOME'],'.config','antgo'))

      shutil.copy(FLAGS.config(), os.path.join(os.environ['HOME'],'.config','antgo','config.xml'))
    except:
      logger.warn('perhaps you want to set default config.xml, please in root authority')
      pass
  else:
    # parse config file
    if not os.path.exists(os.path.join(os.environ['HOME'],'.config','antgo','config.xml')):
      # use default config
      if not os.path.exists(os.path.join(os.environ['HOME'],'.config','antgo')):
        os.makedirs(os.path.join(os.environ['HOME'],'.config','antgo'))

      shutil.copy(os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-1]), 'config.xml'),
                  os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'))

    config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
    Config.parse_xml(config_xml)

  if len(sys.argv) == 1 or sys.argv[1].startswith('-'):
    # interactive control
    shell_process = AntShell(Context(), token)
    shell_process.start()
    return

  if FLAGS.running_platform == 'local' and FLAGS.version() is not None:
    a,b,c = FLAGS.version().split('.')
    sys_a, sys_b, sys_c = version.split('.')
    if int(sys_a) < int(a) or int(sys_b) < int(b) or int(sys_c) < int(c):
      logger.error('antgo version dont satisfy task minimum request (%s)'%FLAGS.version())
      sys.exit(-1)

  # 4.1.step check factory
  factory = getattr(Config, 'factory', None)
  if factory is None or factory == '':
    # give tip
    logger.warn('please antgo -config=... in root authority')

    # plan B
    home_folder = os.environ['HOME']
    Config.data_factory = os.path.join(home_folder, 'antgo', 'dataset')
    Config.task_factory = os.path.join(home_folder, 'antgo', 'task')
    
  if not os.path.exists(Config.data_factory):
    os.makedirs(Config.data_factory)

  if not os.path.exists(Config.task_factory):
    os.makedirs(Config.task_factory)

  task_factory = Config.task_factory
  data_factory = Config.data_factory

  # 6.step parse antgo execute command
  # 6.2.step other command (train, challenge, compose, deploy, server)
  ant_cmd = sys.argv[1]
  if ant_cmd not in _ant_support_commands:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
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
                             tensorflow=True if FLAGS.framework() == 'tensorflow' else False,
                             GAN=True if FLAGS.net() == 'GAN' else False)

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

  if ant_cmd == 'tools/tfgraph':
    import antgo.codebook.tf.tftools as tftools
    tftools.tftool_visualize_pb(FLAGS.tfgraph_path())
    return

  # 7.step check related params
  # 7.1 step check name, if None, set it as current time automatically
  time_stamp = timestamp()
  name = FLAGS.name()
  if name is None:
    logger.error('must set name')
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

  if FLAGS.zoo():
    is_ok = experiment_download_dht(main_folder,
                                    name,
                                    token,
                                    proxy=FLAGS.proxy(),
                                    signature=FLAGS.signature(),
                                    address='http://experiment.mltalker.com/%s.tar.gz'%name)
    if not is_ok:
      logger.error('couldnt load experiment from experiment zoo')
      return

    # swith to experiment in main_folder
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
      logger.error('main executing file dont exist')
      sys.exit(-1)

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
      task_content = '<task><task_name>%s</task_name>' \
                     '<input>' \
                     '<source_data><data_set_name>%s</data_set_name>' \
                     '</source_data>' \
                     '</input>' \
                     '</task>'%('default-task', dataset)
      fp.write(task_content)
    task = os.path.join(task_factory, '%s.xml'%name)

  # 8.2 step load ant context
  ant_context = None
  if main_file is not None and main_file != '':
    ant_context = main_context(main_file, main_folder)
  else:
    ant_context = Context()
  
  # 8.3 step load model config
  main_param = FLAGS.main_param()
  if main_param is not None:
    main_config_path = os.path.join(main_folder, main_param)
    params = yaml.load(open(main_config_path, 'r'))
    ant_context.params = params

  ant_context.name = name
  ant_context.data_factory = data_factory
  ant_context.task_factory = task_factory

  # 8.4 step load experiment
  if FLAGS.from_experiment() is not None and \
          FLAGS.running_platform() == 'local' and \
          ant_cmd not in ['release']:
    experiment_path = os.path.join(dump_dir, FLAGS.from_experiment())
    # if not os.path.exists(experiment_path):
    #   process = multiprocessing.Process(target=experiment_download_dht,
    #                                     args=(dump_dir, FLAGS.from_experiment(), token, token))
    #   process.start()
    #   process.join()

    if not os.path.exists(experiment_path):
      logger.error('couldnt find experiment %s'%FLAGS.from_experiment())
      exit(-1)

    if os.path.exists(os.path.join(dump_dir, FLAGS.from_experiment(), 'train')):
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'train')
    else:
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'inference')
  
  # tools
  if ant_cmd == 'tools/tffrozen':
    # tensorflow tools
    import antgo.codebook.tf.tftools as tftools
    tftools.tftool_frozen_graph(ant_context,
                                dump_dir,
                                time_stamp,
                                FLAGS.tffrozen_input_nodes(),
                                FLAGS.tffrozen_output_nodes())
    return
  elif ant_cmd == 'tools/tfrecords':
    # tensorflow tools
    import antgo.codebook.tf.tftool_records as tftool_records
    tfrecords = \
        tftool_records.AntTFRecords(ant_context,
                                    data_factory,
                                    dataset,
                                    dump_dir)
    tfrecords.start()
    return

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
                              port=FLAGS.port(),
                              time_stamp=time_stamp,
                              devices=FLAGS.devices())

    running_process.start()
  elif ant_cmd == "batch":
    running_process = AntBatch(ant_context,
                               name,
                               token,
                               data_factory,
                               dump_dir,
                               task,
                               unlabel=FLAGS.unlabel(),
                               devices=FLAGS.devices())
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
                                        time_stamp=time_stamp,
                                        running_platform=FLAGS.running_platform(),
                                        max_time=FLAGS.max_time(),
                                        port=FLAGS.port(),
                                        task=FLAGS.task(),
                                        skip_training=FLAGS.skip_training(),
                                        option=FLAGS.option(),
                                        from_experiment=FLAGS.from_experiment(),
                                        devices=FLAGS.devices())
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
                                 data_factory,
                                 dataset,
                                 dump_dir)
    running_process.start()

  # 9.step clear context
  ant_context.wait_until_clear()

if __name__ == '__main__':
  main()
