# encoding=utf-8
# @Time    : 17-5-9
# @File    : challenge.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from antgo.html.html import *
from .base import *
from ..dataflow.common import *
from ..measures.statistic import *
from ..task.task import *
from ..utils import logger
from ..dataflow.recorder import *
import shutil


class AntChallenge(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config=None):
    super(AntChallenge, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_source = ant_data_folder
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config

  def start(self):
    # 0.step loading challenge task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      challenge_task_config = self.rpc("TASK-CHALLENGE")
      if challenge_task_config is None:
        logger.error('couldnt load challenge task')
        exit(0)
      elif challenge_task_config['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        logger.error('prohibit submit challenge task frequently')
        exit(0)
      elif challenge_task_config['status'] == 'UNAUTHORIZED':
        # unauthorized submit challenge task
        logger.error('unauthorized submit challenge task')
        exit(0)
      elif challenge_task_config['status'] == 'OK':
        challenge_task = create_task_from_json(challenge_task_config)
        if challenge_task is None:
          logger.error('couldnt load challenge task')
          exit(0)
        running_ant_task = challenge_task

    if running_ant_task is None:
      # 0.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(0)
      running_ant_task = custom_task

    assert(running_ant_task is not None)

    # 1.step loading test dataset
    logger.info('loading test dataset %s'%running_ant_task.dataset_name)
    ant_test_dataset = running_ant_task.dataset('test',
                                                 os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                 running_ant_task.dataset_params)
    
    with safe_recorder_manager(ant_test_dataset):
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
  
      self.stage = "INFERENCE"
      logger.info('start infer process')
      now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(self.time_stamp))
      infer_dump_dir = os.path.join(self.ant_dump_dir, now_time, 'inference')
      if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)
      else:
        shutil.rmtree(infer_dump_dir)
        os.makedirs(infer_dump_dir)
      
      with safe_recorder_manager(self.context.recorder):
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), infer_dump_dir)

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(ant_test_dataset.size)
  
      self.stage = 'EVALUATION'
      logger.info('start evaluation process')
      evaluation_measure_result = []
  
      with safe_recorder_manager(RecordReader(infer_dump_dir)) as record_reader:
        for measure in running_ant_task.evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          evaluation_measure_result.append(result)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result
      
      logger.info('generate model evaluation report')
      # performace statistic
      everything_to_html(task_running_statictic, os.path.join(self.ant_dump_dir, now_time))

      # compare statistic
      logger.info('start compare process')

      # notify
      self.context.job.send({'DATA': {'STATISTIC': task_running_statictic}})
