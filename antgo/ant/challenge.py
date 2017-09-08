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
      
      # notify
      self.context.job.send({'DATA': {'REPORT': task_running_statictic}})
      
      # Challenge Analysis
      task_running_statictic_mirror = copy.deepcopy(task_running_statictic)
      # compare statistic
      logger.info('start compare process')
      related_model_result = None
      # finding all related model result from server
      
      # deep analysis
      logger.info('deep analysis')
      # finding all reltaed model result from server

      for measure_result in task_running_statictic_mirror[self.ant_name]['measure']:
        if 'info' in measure_result:
          measure_name = measure_result['statistic']['name']
          measure_data = measure_result['info']
          
          if 'analysis' not in task_running_statictic_mirror[self.ant_name]:
            task_running_statictic_mirror[self.ant_name]['analysis'] = {}
          task_running_statictic_mirror[self.ant_name]['analysis'][measure_name] = {}
          
          method_samples_matrix = []
          if related_model_result is not None:
            pass
          else:
            # simple process
            dd = None
            if 'is_correct' in measure_data[0]:
              dd = sorted(measure_data, key=lambda x: x['is_correct'])
            else:
              dd = sorted(measure_data, key=lambda x: x['score'])
            method_samples_matrix.append({'name': self.ant_name, 'data': dd})
          
          # reorganize data as method score matrix
          method_num = len(method_samples_matrix)
          samples_num = len(method_samples_matrix[0]['data'])
          method_measure_mat = np.zeros((method_num, samples_num))
          samples_id = np.zeros((samples_num), np.uint64)
          is_binary_data = False
          for method_id, method_measure_data in enumerate(method_samples_matrix):
            if method_id == 0:
              # record sample id
              for sample_id, sample in enumerate(method_measure_data['data']):
                samples_id[sample_id] = sample['id']
  
            for sample_id, sample in enumerate(method_measure_data['data']):
              if 'is_correct' in sample:
                is_binary_data = True
              else:
                method_measure_mat[method_id, sample_id] = sample['score']
          
          # record method score matrix
          if is_binary_data:
            method_measure_mat = method_measure_mat.astype(np.uint8)
          task_running_statictic_mirror[self.ant_name]['analysis'][measure_name]['data'] = method_measure_mat.tolist()

          # global group
          abstract_group = _group_measure_data_by_tag(method_measure_mat,
                                                      samples_id,
                                                      is_binary_data,
                                                      ant_test_dataset,
                                                      filter_tag=None)
          task_running_statictic_mirror[self.ant_name]['analysis'][measure_name]['group'] = abstract_group
          
          # group by tag
          tags = getattr(ant_test_dataset, 'tag', None)
          if tags is not None:
            for tag in tags:
              tag_group = _group_measure_data_by_tag(method_measure_mat,
                                                     samples_id,
                                                     is_binary_data,
                                                     ant_test_dataset,
                                                     filter_tag=tag)
              if 'tag-group' not in task_running_statictic_mirror[self.ant_name]['analysis'][measure_name]:
                task_running_statictic_mirror[self.ant_name]['analysis'][measure_name]['tag-group'] = []

              task_running_statictic_mirror[self.ant_name]['analysis'][measure_name]['tag-group'].append((tag, tag_group))
          
      # performace statistic
      logger.info('generate model evaluation report')
      everything_to_html(task_running_statictic_mirror, os.path.join(self.ant_dump_dir, now_time))


def _group_measure_data_by_tag(method_score_mat, samples_id, is_binary, data_source, filter_tag=None):
  # group data
  method_num = method_score_mat.shape[0]
  sample_num = method_score_mat.shape[1]
  
  if is_binary:
    # six group (95%, 52%, 42%, 13%, only best, 0%)
    pass
  else:
    # four group (10%, 30%, 50%, 80%)
    pass
  
    # id = ss['id']
    # data = None
    # label = None
    # if filter_tag is not None:
    #   data, label = data_source.at(id)
    #   if filter_tag not in label.tag:
    #     continue
    #
    # if 'is_correct' in ss:
    #   # six group (95%, 52%, 42%, 13%, only best, 0%)
    #   pass
    # else:
    #   # four group (10%, 30%, 50%, 80%)
    #
    #   pass
  return None