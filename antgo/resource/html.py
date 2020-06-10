# encoding=utf-8
# @Time    : 17-5-8
# @File    : resource.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import numpy as np
import base64
import copy
from antgo.utils.encode import *
from jinja2 import Environment, FileSystemLoader
import scipy.misc
import sys
import uuid
if sys.version > '3':
  PY3 = True
else:
  PY3 = False
 
  
PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'templates')),
    trim_blocks=False)


def render_template(template_filename, context):
  return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def everything_to_html(data, dump_dir):
  assert(len(data) == 1)
  # list all statistics
  everything_statistics = []
  model_deep_analysis = []
  model_sig_diffs = []
  for ant_name, ant_info in data.items():
    # 0.step cpu statistic
    if 'cpu' in ant_info:
      cpu_model = ant_info['cpu']['cpu_model']
      cpu_max_util = ant_info['cpu']['cpu_max_usage']
      cpu_mean_util = ant_info['cpu']['cpu_mean_usage']
      cpu_median_util = ant_info['cpu']['cpu_median_usage']

      cpu_statistic = {'statistic': {'name': 'cpu',
                                     'value': [
                                         {'name': cpu_model,
                                          'value': [['cpu', 'max util', 'mean util','median util'],
                                                    ['-', cpu_max_util, cpu_mean_util,cpu_median_util]],
                                          'type': "TABLE"}]}}
      everything_statistics.append(cpu_statistic)

      # 1.step memory statistic
      mem_max_util = ant_info['cpu']['mem_max_usage']
      mem_mean_util = ant_info['cpu']['mem_mean_usage']
      mem_median_util = ant_info['cpu']['mem_median_usage']

      memory_statistic = {'statistic': {'name': 'cpu memory',
                                        'value': [
                                            {'name': 'memory',
                                             'value': [['memory', 'max util', 'mean util', 'median util'],
                                                       ['-', mem_max_util, mem_mean_util, mem_median_util]],
                                             'type': "TABLE"}]}}
      everything_statistics.append(memory_statistic)
    
    # 1.step gpu statistic
    if 'gpu' in ant_info:
      gpu_model = ant_info['gpu']['gpu_model']
      gpu_max_util = ant_info['gpu']['gpu_max_usage']
      gpu_mean_util = ant_info['gpu']['gpu_mean_usage']
      gpu_median_util = ant_info['gpu']['gpu_median_usage']
      
      value_list = [['gpu', 'max util', 'mean util','median util']]
      for gpu_i in range(len(gpu_max_util)):
        value_list.append(['%d'%gpu_i,
                           '%0.4f'%gpu_max_util[gpu_i],
                           '%0.4f'%gpu_mean_util[gpu_i],
                           '%0.4f'%gpu_median_util[gpu_i]])
      
      gpu_statistic = {'statistic': {'name': 'gpu',
                                     'value': [
                                       {'name': gpu_model,
                                        'value': value_list,
                                        'type': "TABLE"}]}}
      everything_statistics.append(gpu_statistic)
      
      gpu_mem_max_util = ant_info['gpu']['gpu_mem_max_usage']
      gpu_mem_mean_util = ant_info['gpu']['gpu_mem_mean_usage']
      gpu_mem_median_util = ant_info['gpu']['gpu_mem_median_usage']
      
      value_list = [['memory', 'max util', 'mean util', 'median util']]
      for gpu_i in range(len(gpu_mem_max_util)):
        value_list.append(['%d'%gpu_i,
                           '%0.4f'%gpu_mem_max_util[gpu_i],
                           '%0.4f'%gpu_mem_mean_util[gpu_i],
                           '%0.4f'%gpu_mem_median_util[gpu_i]])
      
      gpu_memory_statistic = {'statistic': {'name': 'memory',
                                            'value': [
                                              {'name': 'gpu memory',
                                               'value': value_list,
                                               'type': "TABLE"}]}}
      everything_statistics.append(gpu_memory_statistic)
    
    # 2.step time statistic
    if 'time' in ant_info:
      elapsed_time_per_sample = '-'
      if 'elapsed_time_per_sample' in ant_info['time']:
          elapsed_time_per_sample = ant_info['time']['elapsed_time_per_sample']
      time_statistic = {'statistic': {'name': 'time',
                                      'value': [
                                          {'name': 'time',
                                           'value': [['time', 'total time', 'per sample time'],
                                                     ['-', ant_info['time']['elapsed_time'], elapsed_time_per_sample]],
                                           'type': "TABLE"}]}}
      everything_statistics.append(time_statistic)

    # 3.step model measures
    if 'measure' in ant_info:
      for ms in ant_info['measure']:
        everything_statistics.append(ms)

    # 4.step model analysis
    if 'analysis' in ant_info:
      model_analysis = ant_info['analysis']
      for measure_name, measure_data in model_analysis.items():
        for analysis_title, analysis_data in measure_data.items():
          if type(analysis_data) == list:
            # for group
            for tag, tag_data in analysis_data:
              model_deep_analysis.append({'analysis_name': measure_name,
                                          'analysis_tag': analysis_title+'-'+tag,
                                          'analysis_data': tag_data})
          else:
            # for global
            model_deep_analysis.append({'analysis_name': measure_name,
                                        'analysis_tag': analysis_title,
                                        'analysis_data': analysis_data})
    
    # 5.step model significant difference
    if 'significant_diff' in ant_info:
      sig_diff_dict = ant_info['significant_diff']
      # x in significant diff matrix
      benchmark_model_names = []
      # y in significant diff matrix
      measure_names = []
      # score matrix
      benchmark_model_sig_diff = []

      for measure_name, sig_diff in sig_diff_dict.items():
        # y
        measure_names.append(measure_name)
        if len(benchmark_model_names) == 0:
          # x
          benchmark_model_names = [bn['name'] for bn in sig_diff]
        
        # score (-1, 0, +1) matrix
        benchmark_model_sig_diff.append([bn['score'] for bn in sig_diff])

      benchmark_model_sig_diff = np.array(benchmark_model_sig_diff)
      benchmark_model_sig_diff = benchmark_model_sig_diff.transpose()
      benchmark_model_sig_diff = benchmark_model_sig_diff.tolist()
      model_sig_diffs.append({'score': benchmark_model_sig_diff,
                              'benchmark': benchmark_model_names,
                              'measure': measure_names})
        
  statistic_visualization = _transform_statistic_to_visualization(everything_statistics)
  analysis_visualization = _transform_analysis_to_visualization(model_deep_analysis, dump_dir)
  model_sig_diffs_visualization = _transform_significant_to_visualization(model_sig_diffs)
  
  # 6.step model eye analysys
  eye={'eye_measures': []}
  for ant_name, ant_info in data.items():
    if 'eye' in ant_info:
      if not os.path.exists(os.path.join(dump_dir, 'eye_analysis')):
        os.makedirs(os.path.join(dump_dir, 'eye_analysis'))
      
      for measure_name, eye_data in ant_info['eye'].items():
        eye_measure = {'NAME': measure_name}
        eye_tags = set()
        eye_samples = []
        for sample_index, sample in enumerate(eye_data):
          sample_id = sample['id']
          sample_category = sample['category']
          sample_score = sample['score']
          sample_tag = sample['tag']
          sample_data = sample['data']
          eye_tags.update(sample_tag)
          
          # 1.step warp to eye sample
          eye_sample = {}
          eye_sample['ID'] = sample_id
          eye_sample['TAGS'] = sample_tag
          eye_sample['SCORE'] = sample_score
          # 2.step warp data to eye sample
          if sample['data_type'] == 'IMAGE':
            if type(sample_data) == str:
              eye_sample['DATA'] = 'data:image/png;base64,%s' % sample_data
            else:
              with open(os.path.join(dump_dir, 'eye_analysis', '%s.png' % str(sample_id)), 'wb') as fp:
                fp.write(sample_data)
              eye_sample['DATA'] = './eye_analysis/%s.png' % str(sample_id)

            eye_sample['DATA_TYPE'] = 'IMAGE'
          elif sample['data_type'] == 'STRING':
            eye_sample['DATA'] = sample_data
            eye_sample['DATA_TYPE'] = 'STRING'
            
          # 3.step warp gt to eye sample
          if sample['category_type'] == 'IMAGE':
            if type(sample_category) == str:
              eye_sample['GT'] = 'data:image/png;base64,%s' % sample_data
            else:
              with open(os.path.join(dump_dir, 'eye_analysis', '%s_gt.png' % str(sample_id)), 'wb') as fp:
                fp.write(sample_category)
              eye_sample['GT'] = './eye_analysis/%s_gt.png' % str(sample_id)
            eye_sample['GT_TYPE'] = 'IMAGE'
          elif sample['category_type'] == 'STRING':
            eye_sample['GT'] = sample_category
            eye_sample['GT_TYPE'] = 'STRING'

          eye_samples.append(eye_sample)
        
        eye_tags_list = []
        for tt_i, tt in enumerate(eye_tags):
          eye_tags_list.append({'NAME': tt, 'ID': tt_i})

        eye_measure['eye_tags'] = eye_tags_list
        eye_measure['eye_samples'] = eye_samples
        
        eye['eye_measures'].append(eye_measure)
  
  context = {
    'measures': statistic_visualization,
    'analysis': analysis_visualization,
    'compare': model_sig_diffs_visualization,
    'eye': eye,
  }

  # to resource
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  with open(os.path.join(dump_dir, 'statistic-report.html'), 'w') as f:
    ss = render_template('statistic-report.html', context)
    if not PY3:
      ss = ss.encode('utf-8')
    f.write(ss)


def _transform_curve_svg_data(data):
  if type(data) != list:
    data = [data]

  reorganized_data = []
  for curve_data in data:
    if type(curve_data) != np.ndarray:
      curve_data = np.array(curve_data)
    
    if curve_data.size == 0:
      reorganized_data.append([])
      continue
      
    x = curve_data[:, 0]
    y = curve_data[:, 1]

    xlist = x.tolist()
    ylist = y.tolist()

    temp = []
    for xv, yv in zip(xlist, ylist):
      temp.append({'x': str(xv), 'y': str(yv)})

    reorganized_data.append(temp)
  return reorganized_data


def _transform_histogram_svg_data(data):
  reorganized_data = []

  if type(data[0]) != list:
    data = [data]
  for his_data in data:
    svg_data = []
    for xv, yv in zip(range(0, len(his_data)), his_data):
      svg_data.append({'x': str(xv), 'y': str(yv)})

    reorganized_data.append(svg_data)

  return reorganized_data


def _transform_image_data(data):
  ss = base64.b64encode(png_encode(data))
  ss = ss.decode('utf-8')
  return ss


def _transform_statistic_to_visualization(statistic_info):
  visualization_statistic_info = []
  for sta_info in statistic_info:
    for index in range(len(sta_info['statistic']['value'])):
      if sta_info['statistic']['value'][index]['type'] == 'CURVE':
        statistic_value = sta_info['statistic']['value'][index]['value']
        reorganized_statistic_value = _transform_curve_svg_data(statistic_value)

        sta_info_cpy = copy.deepcopy(sta_info)
        sta_info_cpy['statistic']['value'] = [sta_info['statistic']['value'][index]]
        sta_info_cpy['statistic']['value'][0]['value'] = reorganized_statistic_value
        visualization_statistic_info.append(sta_info_cpy)
      elif sta_info['statistic']['value'][index]['type'] == 'SCALAR':
        statistic_value = sta_info['statistic']['value'][index]['value']
        if type(statistic_value) == list:
          # histgram
          reorganized_statistic_value = _transform_histogram_svg_data(statistic_value)

          sta_info_cpy = copy.deepcopy(sta_info)
          sta_info_cpy['statistic']['value'] = [sta_info['statistic']['value'][index]]
          sta_info_cpy['statistic']['value'][0]['value'] = reorganized_statistic_value
          sta_info_cpy['statistic']['value'][0]['type'] = 'HISTOGRAM'
          visualization_statistic_info.append(sta_info_cpy)
        else:
          # scalar
          sta_info_cpy = copy.deepcopy(sta_info)
          sta_info_cpy['statistic']['value'] = [sta_info['statistic']['value'][index]]
          sta_info_cpy['statistic']['value'][0]['value'] = str(sta_info_cpy['statistic']['value'][0]['value'])
          if 'interval' in sta_info_cpy['statistic']['value'][0]:
              sta_info_cpy['statistic']['value'][0]['interval'] = str(sta_info_cpy['statistic']['value'][0]['interval'])
          visualization_statistic_info.append(sta_info_cpy)
      elif sta_info['statistic']['value'][index]['type'] == 'IMAGE':
        image_value = sta_info['statistic']['value'][index]['value']
        # image
        # reorganized_statistic_value = _transform_image_data(image_value)
        reorganized_statistic_value = image_value

        sta_info_cpy = copy.deepcopy(sta_info)
        sta_info_cpy['statistic']['value'] = [sta_info['statistic']['value'][index]]
        sta_info_cpy['statistic']['value'][0]['value'] = reorganized_statistic_value
        sta_info_cpy['statistic']['value'][0]['type'] = 'IMAGE'
        visualization_statistic_info.append(sta_info_cpy)

      else:
        # TABLE / MATRIX
        sta_info_cpy = copy.deepcopy(sta_info)
        sta_info_cpy['statistic']['value'] = [sta_info['statistic']['value'][index]]
        sta_info_cpy['statistic']['value'][0]['value'] = str(sta_info_cpy['statistic']['value'][0]['value'])
        visualization_statistic_info.append(sta_info_cpy)

  return visualization_statistic_info


def _transform_analysis_to_visualization(analysis_info, dump_dir):
  analysis_vis_data = []
  # analysis-name, analysis-tag, analysis_data
  for data in analysis_info:
    analysis_name = data['analysis_name']
    analysis_tag = data['analysis_tag']
    analysis_data = data['analysis_data']
    item = {}
    item['name'] = '%s-%s'%(analysis_name, analysis_tag)
    item['value'] = analysis_data['value'].tolist() if type(analysis_data['value']) != list else analysis_data['value']
    if type(analysis_data['x']) == list:
      item['x'] = analysis_data['x']
    else:
      item['x'] = analysis_data['x'].tolist()
    
    if type(analysis_data['y']) == list:
      item['y'] = analysis_data['y']
    else:
      item['y'] = analysis_data['y'].tolist()

    region_samplings = analysis_data['sampling']
    region_samplings_vis = []
    for sampling in region_samplings:
      mmm = []
      for sampling_data in sampling['data']:
        if sampling_data['type'] == 'IMAGE':
          if type(sampling_data['data']) == str:
            mmm.append({'data':'data:image/png;base64,%s'%sampling_data['data'], 'flag': '', 'type': sampling_data['type']})
          else:
            if not os.path.exists(os.path.join(dump_dir, 'statistic_analysis')):
              os.makedirs(os.path.join(dump_dir, 'statistic_analysis'))

            uuid_name = '%s.png' % str(uuid.uuid4())
            with open(os.path.join(dump_dir, 'statistic_analysis', uuid_name), 'wb') as fp:
              fp.write(sampling_data['data'])

            mmm.append(
              {'data': './statistic_analysis/%s'%uuid_name, 'flag': 'image_container', 'type': sampling_data['type']})
        else:
          mmm.append({'data': sampling_data['data'], 'type': sampling_data['type']})

      region_samplings_vis.append({'name': sampling['name'], 'data': mmm})

    item['region_samplings'] = region_samplings_vis
    analysis_vis_data.append(item)

  return analysis_vis_data


def _transform_significant_to_visualization(sig_diffs):
  return sig_diffs

if __name__ == '__main__':
  # experiment 1
  experiment_1_statis = {}
  experiment_1_statis['aa'] = {}
  experiment_1_statis['aa']['time'] = {}
  experiment_1_statis['aa']['time']['elapsed_time'] = 1.2
  experiment_1_statis['aa']['cpu'] = {}
  experiment_1_statis['aa']['cpu']['mem_mean_usage'] = 23
  experiment_1_statis['aa']['cpu']['mem_median_usage'] = 12
  experiment_1_statis['aa']['cpu']['mem_max_usage'] = 44
  experiment_1_statis['aa']['cpu']['cpu_mean_usage'] = 55
  experiment_1_statis['aa']['cpu']['cpu_median_usage'] = 11
  experiment_1_statis['aa']['cpu']['cpu_max_usage'] = 22
  experiment_1_statis['aa']['cpu']['cpu_model'] = 'aabbcc'

  voc_measure = {'statistic': {'name': 'voc',
                               'value': [{'name': 'MAP', 'value': [23.0, 11.0, 12.0], 'type': 'SCALAR', 'x': 'class',
                                          'y': 'Mean Average Precision'},
                                         {'name': 'Mean-MAP', 'value': 0.23, 'type': 'SCALAR'}]}}

  roc_auc_measure = {'statistic': {'name': 'roc_auc',
                                   'value': [{'name': 'ROC', 'value': [[[0,0],[1,1],[2,3]],
                                                                       [[0,3],[2,5],[3,0]]],
                                              'type': 'CURVE', 'x': 'FP', 'y': 'TP', 'legend':['class-0','class-1']},
                                             {'name': 'AUC', 'value': [0.3, 0.4], 'type': 'SCALAR', 'x': 'class',
                                              'y': 'AUC'}]}}

  pr_f1_measure = {'statistic': {'name': 'pr_f1',
                                 'value': [{'name': 'Precision-Recall',
                                            'value': [[[0,0],[1,1],[2,3]],
                                                                       [[0,3],[2,5],[3,0]]],
                                            'type': 'CURVE', 'x': 'precision', 'y': 'recall'},
                                           {'name': 'F1', 'value': [1.0, 2.0], 'type': 'SCALAR', 'x': 'class',
                                            'y': 'F1'}]}}

  confusion_m = {'statistic': {'name': 'cm',
                               'value': [{'name': 'ccmm', 'value': (np.ones((3, 4)) * 3).tolist(), 'type': 'MATRIX',
                                          'x': ['a','b','c','d'], 'y': ['x','y','z']}]}}

  random_img = np.random.random((100,100))
  random_img = random_img * 255
  random_img = random_img.astype(np.uint8)
  image_m = {'statistic': {'name': 'image',
                               'value': [{'name': 'image', 'value': random_img, 'type': 'IMAGE'}]}}

  experiment_1_statis['aa']['measure'] = [voc_measure, roc_auc_measure, pr_f1_measure, confusion_m, image_m]
  experiment_1_statis['aa']['eye'] = {}
  experiment_1_statis['aa']['eye']['MM'] = [{'id': 0, 'category': 'hello', 'score':0.2, 'tag':['A','B'], 'data': 'mma'},
                                            {'id': 3, 'category': 'miao', 'score': 0.4, 'tag':['C'], 'data': 'yYU'},
                                            {'id': 1, 'category': 'world', 'score':1.0, 'tag':['A'], 'data': 'UUI'}]


  # analysis data
  scores = np.random.randint(0,100,200)
  scores = scores.reshape((2,100))

  sample_index = list(range(100))
  image = cv2.imread('/Users/jian/Downloads/owl1.jpg')
  image = image[...,::-1]
  # image = base64.b64encode(png_encode(image))
  # image = image.decode('utf-8')
  hr_samples = []
  mr_samples = []
  lr_samples = []
  for _ in range(10):
    hr_samples.append({'type': 'IMAGE', 'data': image})
    mr_samples.append({'type': 'IMAGE', 'data': image})
    lr_samples.append({'type': 'IMAGE', 'data': image})

  experiment_1_statis['aa']['analysis'] = {}
  experiment_1_statis['aa']['analysis']['AA']={}
  experiment_1_statis['aa']['analysis']['AA']['Global']={
    'value': scores,
    'type': 'MATRIX',
    'x': sample_index,
    'y': ['12345.1231.134134521','20142312.1324.135234'],
    'sampling': [{'name': 'High Score Region', 'data': hr_samples},
                 {'name': 'Middle Score Region', 'data': mr_samples},
                 {'name': 'Low Score Region', 'data': lr_samples}],
  }

  #
  # import cv2
  mm = cv2.imread('/Users/jian/Downloads/owl1.jpg')
  mm = mm[...,::-1]

  experiment_1_statis['aa']['eye']['NN'] = [{'id': 0, 'category': mm, 'score':0.2, 'tag':['A','B'], 'data': mm},
                                            {'id': 3, 'category': mm, 'score': 0.4, 'tag':['C'], 'data': mm},
                                            {'id': 1, 'category': mm, 'score':1.0, 'tag':['A'], 'data': mm}]
  
  # # experiment 2
  # experiment_2_statis = {}
  # experiment_2_statis['aa'] = {}
  # experiment_2_statis['aa']['time'] = {}
  # experiment_2_statis['aa']['time']['elapsed_time'] = 1.2
  # experiment_2_statis['aa']['cpu'] = {}
  # experiment_2_statis['aa']['cpu']['mem_mean_usage'] = 23
  # experiment_2_statis['aa']['cpu']['mem_median_usage'] = 12
  # experiment_2_statis['aa']['cpu']['mem_max_usage'] = 44
  # experiment_2_statis['aa']['cpu']['cpu_mean_usage'] = 55
  # experiment_2_statis['aa']['cpu']['cpu_median_usage'] = 11
  # experiment_2_statis['aa']['cpu']['cpu_max_usage'] = 22
  # experiment_2_statis['aa']['cpu']['cpu_model'] = 'aabbcc'
  #
  # voc_measure = {'statistic': {'name': 'voc',
  #                              'value': [{'name': 'MAP', 'value': [2.0, 1.0, 2.0], 'type': 'SCALAR', 'x': 'class',
  #                                         'y': 'Mean Average Precision'},
  #                                        {'name': 'Mean-MAP', 'value': 0.12, 'type': 'SCALAR'}]}}
  #
  # roc_auc_measure = {'statistic': {'name': 'roc_auc',
  #                                  'value': [
  #                                      {'name': 'ROC',
  #                                       'value': [(np.ones((3, 2)) * 4).tolist(), (np.ones((3, 2)) * 1).tolist()],
  #                                       'type': 'CURVE',
  #                                       'x': 'FP', 'y': 'TP'},
  #                                      {'name': 'AUC', 'value': [0.12, 0.34], 'type': 'SCALAR', 'x': 'class',
  #                                       'y': 'AUC'}]}}
  #
  # pr_f1_measure = {'statistic': {'name': 'pr_f1',
  #                                'value': [
  #                                    {'name': 'Precision-Recall',
  #                                     'value': [(np.ones((4, 2)) * 1).tolist(), (np.ones((4, 2)) * 2).tolist()],
  #                                     'type': 'CURVE', 'x': 'precision', 'y': 'recall'},
  #                                    {'name': 'F1', 'value': [1.4, 1.0], 'type': 'SCALAR', 'x': 'class', 'y': 'F1'}]}}
  #
  # confusion_m = {'statistic': {'name': 'cm',
  #                              'value': [{'name': 'ccmm', 'value': (np.ones((3, 4)) * 8).tolist(), 'type': 'MATRIX',
  #                                         'x': 'class', 'y': 'class'}]}}
  #
  # experiment_2_statis['aa']['measure'] = [voc_measure, roc_auc_measure, pr_f1_measure, confusion_m]

  # ss = multi_repeats_measures_statistic([experiment_1_statis, experiment_2_statis])
  everything_to_html(experiment_1_statis, '/Users/jian/Downloads/')