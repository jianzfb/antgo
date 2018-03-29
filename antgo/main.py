# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime

from antgo import config
from antgo.ant import flags
from antgo.ant.cmd import *
from antgo.ant.generate import *
from antgo.ant.utils import *
from antgo.sandbox.sandbox import *
from antgo.utils.utils import *
from antgo.utils.dht import *
import multiprocessing

if sys.version > '3':
    PY3 = True
else:
    PY3 = False


def _check_environment():
  is_in_mltalker = True if os.environ.get('ANT_ENVIRONMENT', '') != '' else False
  return is_in_mltalker

_ant_support_commands = ["train", "challenge", "deploy", "dataset", "config", "tools/tffrozen", "tools/tfrecords"]

#############################################
#######   antgo parameters            #######
#############################################
flags.DEFINE_string('main_file', None, 'main file')
flags.DEFINE_string('main_param', None, 'model parameters')
flags.DEFINE_string('main_folder', None, 'resource folder')
flags.DEFINE_string('task', None, 'task file')
flags.DEFINE_string('dataset', None, 'dataset')
flags.DEFINE_boolean('public', False, 'public or private')
flags.DEFINE_boolean('local', False, 'cloud or local')
flags.DEFINE_string('dump', None, 'dump dir')
flags.DEFINE_string('token', None, 'token')
flags.DEFINE_string('platform', 'local', 'local or cloud')
flags.DEFINE_string('sandbox_time', None, 'max running time')
flags.DEFINE_string('from_experiment', None, 'load model from experiment')
flags.DEFINE_string('data_factory', None, '')
flags.DEFINE_string('task_factory', None, '')
#############################################
########  tools - tffrozen            #######
#############################################
flags.DEFINE_string('tffrozen_input_nodes', '', 'input node names in graph')
flags.DEFINE_string('tffrozen_output_nodes', '', 'output node names in graph')
#############################################
########  tools - tfrecords           #######
#############################################
flags.DEFINE_string('tfrecords_data_dir', None, 'data folder')
flags.DEFINE_string('tfrecords_label_dir', None, 'label folder')
flags.DEFINE_string('tfrecords_record_dir', None, 'output tfrecord folder')
flags.DEFINE_string('tfrecords_label_suffix', '', 'label suffix')
flags.DEFINE_string('tfrecords_train_or_test', 'train', 'train or test')
flags.DEFINE_string('tfrecords_shards', 10, 'tfrecord shards')

FLAGS = flags.AntFLAGS
Config = config.AntConfig


def main():
  # 1.step antgo logo
  main_logo()

  # 2.step check antgo support command
  if len(sys.argv) == 1:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    sys.exit(-1)

  # 3.step parse antgo running params
  if sys.argv[1].startswith('--') or sys.argv[1].startswith('-'):
    flags.cli_param_flags(sys.argv[1:])
  else:
    flags.cli_param_flags(sys.argv[2:])

  if sys.argv[1] == 'config':
    data_factory = FLAGS.data_factory()
    task_factory = FLAGS.task_factory()
    config_xml = os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-2]), 'config.xml')
    Config.write_xml(config_xml, {'data_factory': data_factory,
                                  'task_factory': task_factory})
    logger.info('success to update config.xml')
    return

  # 4.step load antgo global config
  config_xml = os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-2]), 'config.xml')
  Config.parse_xml(config_xml)

  # 4.1.step check data_factory
  data_factory = getattr(Config, 'data_factory', None)
  if data_factory is None or data_factory == '':
    # give tip
    logger.warn('please antgo config --data_factory=... --data_factory=... in root authority')
    # plan B
    home_folder = os.environ['HOME']
    data_factory = os.path.join(home_folder, 'antgo', 'antgo-dataset')
    Config.data_factory = data_factory

  if not os.path.exists(data_factory):
    os.makedirs(data_factory)

  # 4.2.step check task_factory
  task_factory = getattr(Config, 'task_factory', None)
  if task_factory is None or task_factory == '':
    # give tip
    logger.warn('please antgo config --data_factory=... --data_factory=... in root authority')
    # plan B
    home_folder = os.environ['HOME']
    task_factory = os.path.join(home_folder, 'antgo', 'antgo-task')
    Config.task_factory = task_factory

  if not os.path.exists(task_factory):
    os.makedirs(task_factory)

  # 5.step parse antgo running token (secret)
  token = FLAGS.token()
  if not PY3 and token is not None:
    token = unicode(token)

  # 6.step parse antgo execute command
  # 6.1.step interactive control context
  if sys.argv[1].startswith('--') or sys.argv[1].startswith('-'):
    # interactive control
    cmd_process = AntCmd(token)
    cmd_process.start()
    return

  # 6.2.step other command (train, challenge, compose, deploy, server)
  ant_cmd = sys.argv[1]
  if ant_cmd not in _ant_support_commands:
    logger.error('antgo cli support( %s )command'%",".join(_ant_support_commands))
    return

  # 7.step check related params
  # 7.1 step check name, if None, set it as current time automatically
  time_stamp = timestamp()
  name = datetime.fromtimestamp(time_stamp).strftime('%Y%m%d.%H%M%S.%f')
  if not PY3:
    name = unicode(name)
  
  # 7.2 check main folder (all related model code, includes main_file and main_param)
  main_folder = FLAGS.main_folder()
  if main_folder is None:
    main_folder = os.path.abspath(os.curdir)
  
  # 7.3 check dump dir (all running data is stored here)
  dump_dir = FLAGS.dump()
  if dump_dir is None:
    dump_dir = os.path.join(os.path.abspath(os.curdir), 'dump')
    if not os.path.exists(dump_dir):
      os.makedirs(dump_dir)

  # 7.4.step tools (option)
  if ant_cmd == 'tools/tffrozen':
    # tensorflow tools
    import antgo.tools.tftools as tftools
    tftools.tftool_frozen_graph(dump_dir,
                                FLAGS.from_experiment(),
                                FLAGS.tffrozen_input_nodes(),
                                FLAGS.tffrozen_output_nodes())
    return
  elif ant_cmd == 'tools/tfrecords':
    # tensorflwo tools
    import antgo.tools.tftools as tftools
    tftools.tftool_generate_image_records(FLAGS.tfrecords_data_dir,
                                          FLAGS.tfrecords_label_dir,
                                          FLAGS.tfrecords_record_dir,
                                          FLAGS.tfrecords_label_suffix,
                                          FLAGS.tfrecords_train_or_test,
                                          FLAGS.tfrecords_shards)
    return

  # 7.5 check main file
  main_file = FLAGS.main_file()
  if main_file is None or not os.path.exists(os.path.join(main_folder, main_file)):
    logger.error('main executing file dont exist')
    sys.exit(-1)

  # 8 step ant running
  # 8.1 step what is task
  task = FLAGS.task()
  if task is not None:
    task = os.path.join(task_factory, task)
  dataset = FLAGS.dataset()
  if task is None and dataset is not None:
    # build default task
    with open(os.path.join(task_factory, '%s.xml'%name), 'w') as fp:
      task_content = '<task><task-name>%s</task-name>' \
                     '<input>' \
                     '<source_data><data_set_name>%s</data_set_name>' \
                     '</source_data>' \
                     '</input>' \
                     '</task>'%('default-task', dataset)
      fp.write(task_content)
    task = os.path.join(task_factory, '%s.xml'%name)
  
  # 8.2 step load ant context
  ant_context = main_context(main_file, main_folder)
  if FLAGS.from_experiment() is not None:
    experiment_path = os.path.join(dump_dir, FLAGS.from_experiment())
    if not os.path.exists(experiment_path):
      process = multiprocessing.Process(target=experiment_download_dht,
                                        args=(dump_dir, FLAGS.from_experiment(), token, token))
      process.start()
      process.join()

    if not os.path.exists(experiment_path):
      logger.error('couldnt find experiment %s'%FLAGS.from_experiment())
      exit(-1)

    if os.path.exists(os.path.join(dump_dir, FLAGS.from_experiment(), 'train')):
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'train')
    else:
      ant_context.from_experiment = os.path.join(dump_dir, FLAGS.from_experiment(), 'inference')

  # 8.3 step load model config
  main_param = FLAGS.main_param()
  if main_param is not None:
    main_config_path = os.path.join(main_folder, main_param)
    params = yaml.load(open(main_config_path, 'r'))
    ant_context.params = params
  
  if ant_cmd == "train":
    sandbox_time = FLAGS.sandbox_time()
    with running_sandbox(sandbox_time=sandbox_time,
                         sandbox_dump_dir=dump_dir,
                         sandbox_experiment=name,
                         sandbox_user_token=token):
      running_process = AntTrain(ant_context,
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
  elif ant_cmd == 'challenge':
    running_process = AntChallenge(ant_context,
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
  elif ant_cmd == "deploy":
    pass
  elif ant_cmd == 'dataset':
    running_process = AntGenerate(ant_context,
                                  name,
                                  data_factory,
                                  dump_dir,
                                  token,
                                  dataset,
                                  FLAGS.public(),
                                  FLAGS.local())
    running_process.start()

  # 9.step clear context
  ant_context.wait_until_clear()

if __name__ == '__main__':
  main()
