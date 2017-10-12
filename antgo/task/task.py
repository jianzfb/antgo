from __future__ import unicode_literals
from __future__ import division
from antgo.dataflow.dataset.standard import Standard
from antgo.measures.measure import *
from antgo.dataflow.dataset import *
from antgo.utils import logger
import json
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET


class AntTask(object):
  def __init__(self, task_id, task_name, task_type_id, task_type,
               dataset_id, dataset_name, dataset_params,
               estimation_procedure_type, estimation_procedure_params,
               evaluation_measure, cost_matrix,
               dataset_url=None,
               class_label=None,
               ext_params=None,
               ant_context=None):
    self._task_id = task_id
    self._task_name = task_name
    self._task_type_id = task_type_id
    self._task_type = task_type
    self._dataset_id = dataset_id
    self._dataset_name = dataset_name
    self._dataset_params = dataset_params
    self._dataset_url = dataset_url
    self._estimation_procedure_type = estimation_procedure_type
    self._estimation_procedure_params = estimation_procedure_params
    self._evaluation_measure = evaluation_measure
    self._cost_matrix = cost_matrix
    self._ant_context = ant_context

    if self._dataset_params is None:
      self._dataset_params = {}
    if self._dataset_url is not None and len(self._dataset_url) > 0:
      self._dataset_params['dataset_url'] = self._dataset_url

    # dataset class
    if self._dataset_name is not None:
      if self._task_id == -1 and \
                      self._ant_context is not None and \
                      self._ant_context.dataset_factory is not None:
          # get dataset IO from custom dataset factory
          # logger.info('dataset io from custom dataset factory')
          self._ant_dataset = self._ant_context.dataset_factory(self._dataset_name)
      else:
          # get dataset IO
          # logger.info('dataset io from mltalker')
          self._ant_dataset = AntDataset(self._dataset_name)

    # related evaluation measures
    self._ant_measures = AntMeasures(self)
    self._class_label = class_label

    # config extent params
    if ext_params is not None:
      for k, v in ext_params.items():
        if k != 'self':
          setattr(self, k, v)

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def dataset_params(self):
    '''

    Returns
    -------

    '''
    return self._dataset_params

  @property
  def dataset(self):
    return self._ant_dataset

  @property
  def task_name(self):
    return self._task_name

  @property
  def estimation_procedure(self):
    '''

    Returns
    -------
    return how to estimate ml model
    '''
    return self._estimation_procedure_type

  @property
  def estimation_procedure_params(self):
    '''

    Returns
    -------
    return estimation procedure parameters
    '''
    return self._estimation_procedure_params

  @property
  def evaluation_measures(self):
    '''

    Returns
    -------
    return how to evaluate ml model
    '''
    return self._ant_measures.measures(self._evaluation_measure)

  def evaluation_measure(self, measure_name):
    '''

    :param measure_name:
    :return:
    '''
    return self._ant_measures.measures(measure_name)

  @property
  def cost_matrix(self):
    '''

    Returns
    -------
    return how to punish prediciton result
    '''
    return self._cost_matrix

  @property
  def class_label(self):
    return self._class_label

  @property
  def task_id(self):
    return self._task_id

  @property
  def task_type(self):
    return self._task_type

  @staticmethod
  def support_task_types():
    return ['OBJECT-DETECTION', 'SEGMENTATION', 'CLASSIFICATION', 'REGRESSION', 'INSTANCE-SEGMENTATION', 'MATTING']


def create_dummy_task(task_type):
  return AntTask(task_id=-1,
                 task_name=None,
                 task_type_id=-1,
                 task_type=task_type,
                 dataset_id=-1,
                 dataset_name='',
                 dataset_params=None,
                 estimation_procedure_type='',
                 estimation_procedure_params=None,
                 evaluation_measure=None,
                 cost_matrix=None)


def create_task_from_json(task_config_json, ant_context=None):
  try:
    # 1.step about task basic
    task = task_config_json['task']
    task_id = task['task_id']
    task_name = task['task_name']
    task_type_id = task['task_type_id']
    task_type = task['task_type']
    task_params = task['task_params']

    # 2.step about task input
    dataset_id = -1
    dataset_name = ""
    target_feature = ""
    dataset_params = {}
    dataset_url = ""
    estimation_procedure_type = ""
    inputs = task['input']
    estimation_procedure_params = {}
    task_cost_matrix = []
    task_evaluation_measures = []
    class_label = None
    task_ext_params = task_params
    for term in inputs:
      if term['name'] == 'source_data':
        dataset_id = term['data_set']['data_set_id']
        dataset_name = term['data_set']['data_set_name']
        if 'data_set_params' in term['data_set']:
          dataset_params = term['data_set']['data_set_params']
        if 'data_set_url' in term['data_set']:
          dataset_url = term['data_set']['data_set_url']
      elif term['name'] == 'estimation_procedure':
        estimation_procedure_type = term['estimation_procedure']['type']
        estimation_procedure_params = {}
        for kk in term['estimation_procedure']['parameter']:
          estimation_procedure_params[kk['name']] = kk['value']
      elif term['name'] == 'cost_matrix':
        task_cost_matrix = term['cost_matrix']
      elif term['name'] == 'evaluation_measures':
        task_evaluation_measures = []
        for measure_name, measure_param in term['evaluation_measures']['evaluation_measure']:
          task_evaluation_measures.append(measure_name)
          if measure_param is not None and len(measure_param) > 0:
            # params = {}
            for k, v in measure_param.items():
              # params[k.strip()] = v
              if k.strip() not in task_ext_params:
                task_ext_params[k.strip()] = v
            # task_ext_params.update(params)
      elif term['name'] == 'info':
        if 'class_label' in term:
          class_label = term['class_label']
      elif term['name'] == 'ext':
        term.pop('name')
        task_ext_params.update(term)

    return AntTask(task_id=task_id,
                   task_name=task_name,
                   task_type_id=task_type_id,
                   task_type=task_type,
                   dataset_id=dataset_id,
                   dataset_name=dataset_name,
                   dataset_url=dataset_url,
                   dataset_params=dataset_params,
                   estimation_procedure_type=estimation_procedure_type,
                   estimation_procedure_params=estimation_procedure_params,
                   evaluation_measure=task_evaluation_measures,
                   cost_matrix=task_cost_matrix,
                   class_label=class_label,
                   ext_params=task_ext_params,
                   ant_context=ant_context)
  except:
    return None


def create_task_from_xml(task_config_xml, ant_context):
  try:
    task_name = None
    task_type = None
    dataset_name = None
    dataset_params = {}
    task_ext_params = {}
    estimation_procedure_type = None
    task_evaluation_measures = []
    class_label = []
    estimation_procedure_params = {}
    tree = ET.ElementTree(file=task_config_xml)
    root = tree.getroot()
    for child in root:
      if child.tag == 'task_name':
        task_name = child.text.strip()
      elif child.tag == 'task_type':
        if child.text is not None:
          task_type = child.text.strip()
      elif child.tag == 'input':
        for input_item in child:
          if input_item.tag == 'source_data':
            for data_item in input_item:
              if data_item.tag == 'data_set_name':
                dataset_name = data_item.text.strip()
              elif data_item.tag == 'data_set_params':
                for parameter in data_item:
                  key = parameter[0].text.strip() if parameter[0].tag == 'name' else parameter[1].text.strip()
                  value = parameter[0].text.strip() if parameter[0].tag == 'value' else parameter[1].text.strip()
                  dataset_params[key] = value
          elif input_item.tag == 'estimation_procedure':
            for data_item in input_item:
              if data_item.tag == 'type':
                if data_item.text is not None:
                  estimation_procedure_type = data_item.text.strip()
              elif data_item.tag == 'parameter':
                if data_item[0].text is not None and \
                                data_item[1].text is not None:
                  key = data_item[0].text.strip() if data_item[0].tag == 'name' else data_item[1].text.strip()
                  value = data_item[0].text.strip() if data_item[0].tag == 'value' else data_item[1].text.strip()
                  estimation_procedure_params[key] = value
          elif input_item.tag == 'evaluation_measures':
            for data_item in input_item:
              if data_item.tag == 'evaluation_measure':
                task_evaluation_measures.append(data_item.text.strip())
                if len(data_item.attrib) > 0:
                  params = {}
                  for k,v in data_item.attrib.items():
                    params[data_item.text.strip()+'_'+k.strip()] = v
                  task_ext_params.update(params)
          elif input_item.tag == 'info':
            for data_item in input_item:
              if data_item.tag == 'class_label':
                for c in data_item:
                    class_label.append(c.text.strip())
      elif child.tag == 'output':
        for input_item in child:
          if input_item.tag == 'generate_report':
            task_ext_params.update({'is_generate_report':int(input_item.text.strip())})

    return AntTask(task_id=-1,
                   task_name=task_name,
                   task_type_id=-1,
                   task_type=task_type,
                   dataset_id=-1,
                   dataset_name=dataset_name,
                   dataset_params=dataset_params,
                   estimation_procedure_type=estimation_procedure_type,
                   estimation_procedure_params=estimation_procedure_params,
                   evaluation_measure=task_evaluation_measures,
                   cost_matrix=None,
                   class_label=class_label,
                   ext_params=task_ext_params,
                   ant_context=ant_context)
  except:
    return None