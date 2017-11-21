# encoding=utf-8
# @Time    : 17-5-8
# @File    : html.py
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

      memory_statistic = {'statistic': {'name': 'memory',
                                        'value': [
                                            {'name': 'memory info',
                                             'value': [['memory', 'max util', 'mean util', 'median util'],
                                                       ['-', mem_max_util, mem_mean_util, mem_median_util]],
                                             'type': "TABLE"}]}}
      everything_statistics.append(memory_statistic)

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

      model_sig_diffs.append({'score': benchmark_model_sig_diff,
                              'benchmark': benchmark_model_names,
                              'measure': measure_names})
        
  statistic_visualization = _transform_statistic_to_visualization(everything_statistics)
  analysis_visualization = _transform_analysis_to_visualization(model_deep_analysis)
  model_sig_diffs_visualization = _transform_significant_to_visualization(model_sig_diffs)
  
  context = {
    'measures': statistic_visualization,
    'analysis': analysis_visualization,
    'compare': model_sig_diffs_visualization
  }

  # to html
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
        reorganized_statistic_value = _transform_image_data(image_value)

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


def _transform_analysis_to_visualization(analysis_info):
  analysis_vis_data = []
  # analysis-name, analysis-tag, analysis_data
  for data in analysis_info:
    analysis_name = data['analysis_name']
    analysis_tag = data['analysis_tag']
    analysis_data = data['analysis_data']
    item = {}
    item['name'] = '%s-%s'%(analysis_name, analysis_tag)
    item['value'] = analysis_data['value'].tolist()
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
      region_samplings_vis.append({'name': sampling['name'], 'data': sampling['data']})

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
  everything_to_html(experiment_1_statis, '/Users/zhangken/Downloads/')