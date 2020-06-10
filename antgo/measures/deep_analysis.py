# -*- coding: UTF-8 -*-
# @Time    : 17-9-19
# @File    : deep_analysis.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.utils._resize import *
from antgo.utils.encode import *
import base64
import copy

# TODO: update incoming (data relevant)
def _whats_data(data_source, id, infos):
  d = data_source.at(id)
  if type(d) == tuple or type(d) == list:
    d = d[0]
  
  if data_source.dataset_type == "TEXT":
    # text
    return {'type': 'TEXT', 'data': d}
  elif data_source.dataset_type == "IMAGE":
    # image
    png_data = png_encode(d, True)
    return {'type': 'IMAGE', 'data': base64.b64encode(png_data).decode('utf-8')}
  
  return None

def discrete_multi_model_measure_analysis(samples_score, samples_map, data_source, filter_tag=None, random_sampling=5):
  # 95%, 52%, 42%, 13%, only best, 0%
  # correct is 1; error is 0
  # 95% - sample could be recognized correctly by 95% models
  # 52% - sample could be recognized correctly by 52% models
  # ...
  # filter by tag
  max_num = samples_score.shape[1]
  assert(max_num == len(samples_map))

  remained_id = []
  if filter_tag is not None:
    for data_index in range(max_num):
      _, label = data_source.at(samples_map[data_index]['id'])
      if filter_tag in label['tag']:
        remained_id.append(data_index)

  if len(remained_id) == 0:
    remained_id = range(max_num)

  samples_score = samples_score[:, remained_id]
  model_num, samples_num = samples_score.shape[0:2]

  # sort cols
  cols_scores = np.sum(samples_score, axis=0)
  cols_orders = np.argsort(-cols_scores)    # decending
  ordered_samples_score = samples_score[:, cols_orders]
  ordered_data_id = [samples_map[remained_id[i]]['id'] for i in cols_orders]

  # sort rows
  rows_scores = np.sum(ordered_samples_score, axis=1)
  rows_orders = np.argsort(-rows_scores)    # decending
  ordered_samples_score = ordered_samples_score[rows_orders, :]
  ordered_model_id = [i for i in rows_orders]

  score_hist = np.sum(ordered_samples_score, axis=0)
  score_hist = score_hist / float(model_num)
  # 95% (94% ~ 96%)
  pos_95 = np.searchsorted(-score_hist, [-0.95])[0]
  pos_95_s = int(np.maximum(0, pos_95 - random_sampling/2))
  pos_95_e = int(np.minimum(samples_num, pos_95_s+random_sampling))

  if pos_95_e <= pos_95_s:
    pos_95_e = pos_95_s + 1
  region_95 = cols_orders[pos_95_s: pos_95_e]
  region_95 = [_whats_data(data_source,
                           samples_map[remained_id[index]]['id'],
                           samples_map[remained_id[index]]) for index in region_95]

  # 52%
  pos_52 = np.searchsorted(-score_hist, [-0.52])[0]
  pos_52_s = int(np.maximum(0, pos_52 - random_sampling/2))
  pos_52_e = int(np.minimum(samples_num, pos_52_s+random_sampling))
  if pos_52_e <= pos_52_s:
    pos_52_e = pos_95_s + 1
  region_52 = cols_orders[pos_52_s:pos_52_e]
  region_52 = [_whats_data(data_source,
                           samples_map[remained_id[index]]['id'],
                           samples_map[remained_id[index]]) for index in region_52]

  # 42%
  pos_42 = np.searchsorted(-score_hist, [-0.42])[0]
  pos_42_s = int(np.maximum(0, pos_42 - random_sampling/2))
  pos_42_e = int(np.minimum(samples_num, pos_42_s+random_sampling))
  if pos_42_e <= pos_42_s:
    pos_42_e = pos_42_s + 1
  # region_42 = np.arange(pos_42_s, pos_42_e)
  region_42 = cols_orders[pos_42_s:pos_42_e]
  region_42 = [_whats_data(data_source,
                           samples_map[remained_id[index]]['id'],
                           samples_map[remained_id[index]]) for index in region_42]
  
  # 13%
  pos_13 = np.searchsorted(-score_hist, [-0.13])[0]
  pos_13_s = int(np.maximum(0, pos_13 - random_sampling/2))
  pos_13_e = int(np.minimum(samples_num, pos_13_s+random_sampling))
  if pos_13_e <= pos_13_s:
    pos_13_e = pos_13_s + 1
  region_13 = cols_orders[pos_13_s: pos_13_e]
  # region_13 = np.arange(pos_13_s, pos_13_e)
  region_13 = [_whats_data(data_source,
                           samples_map[remained_id[index]]['id'],
                           samples_map[remained_id[index]]) for index in region_13]

  # only best
  pos_one = np.searchsorted(-score_hist, [-1.0])[0]
  pos_one_s = int(np.maximum(0, pos_one - random_sampling/2))
  pos_one_e = int(np.minimum(samples_num, pos_one_s+random_sampling))
  if pos_one_e <= pos_one_s:
    pos_one_e = pos_one_s + 1
  # region_one = np.arange(pos_one_s,pos_one_e)
  region_one = cols_orders[pos_one_s: pos_one_e]
  region_one = [_whats_data(data_source,
                            samples_map[remained_id[index]]['id'],
                            samples_map[remained_id[index]]) for index in region_one]

  # 0%
  pos_zero = np.searchsorted(-score_hist, [0.0])[0]
  pos_zero_s = int(np.maximum(0, pos_zero - random_sampling / 2))
  pos_zero_e = int(np.minimum(samples_num, pos_zero_s + random_sampling))
  if pos_zero_e <= pos_zero_s:
    pos_zero_e = pos_zero_s + 1
  region_zero = cols_orders[pos_zero_s: pos_zero_e]
  # region_zero = np.arange(pos_zero_s, pos_zero_e)
  region_zero = [_whats_data(data_source,
                            samples_map[remained_id[index]]['id'],
                            samples_map[remained_id[index]]) for index in region_zero]

  return ordered_samples_score.tolist(), \
         ordered_model_id, \
         ordered_data_id, \
         region_95, \
         region_52, \
         region_42, \
         region_13, \
         region_one, \
         region_zero


def continuous_multi_model_measure_analysis(samples_score, samples_map, data_source, filter_tag=None, random_sampling=10):
  max_num = samples_score.shape[1]
  assert(max_num == len(samples_map))
  # filter by tag
  remained_id = []
  if filter_tag is not None:
    for data_index in range(max_num):
      _, label = data_source.at(samples_map[data_index]['id'])
      if filter_tag == label['tag']:
        remained_id.append(data_index)
  
  if len(remained_id) == 0:
    remained_id = range(max_num)
  
  samples_score = samples_score[:, remained_id]
  
  # low,  middle,  high
  model_num, samples_num = samples_score.shape[0:2]

  # reorder cols (samples)
  model_score = np.sum(samples_score, axis=0)
  reorganized_sample_id = np.argsort(-model_score)    # decending
  output_reorganized_sample_id = [samples_map[remained_id[i]]['id'] for i in reorganized_sample_id]
  reorganized_samples_score = samples_score[:, reorganized_sample_id]
  samples_score = reorganized_samples_score.copy()

  # reorder rows (models)
  s = np.sum(reorganized_samples_score, axis=1)
  reorganized_model_id = np.argsort(-s)               # decending
  reorganized_samples_score = samples_score[reorganized_model_id, :]

  # high score region (0 ~ 1/10) - good
  region_start = 0
  region_end = int(np.minimum(samples_num / 10 * 1, samples_num))
  if region_end == 0:
    region_end = 1
  high_region_sampling = reorganized_sample_id[region_start:region_end]
  high_region_random_sampling = np.minimum(len(high_region_sampling), random_sampling)
  high_region_sampling = np.random.choice(high_region_sampling, high_region_random_sampling, False)
  high_region_sampling = [_whats_data(data_source,
                                      samples_map[remained_id[index]]['id'],
                                      samples_map[remained_id[index]]) for index in high_region_sampling]
  
  # middle score region (4/10 ~ 6/10) - just so so
  region_start = int(np.maximum(samples_num / 10 * 4, 0))
  region_end = int(np.minimum(samples_num / 10 * 6, samples_num))
  if region_end == 0:
    region_end = 1
  middle_region_sampling = reorganized_sample_id[region_start: region_end]
  middle_region_random_sampling = np.minimum(len(middle_region_sampling), random_sampling)
  middle_region_sampling = np.random.choice(middle_region_sampling, middle_region_random_sampling, False)
  middle_region_sampling = [_whats_data(data_source,
                                        samples_map[remained_id[index]]['id'],
                                        samples_map[remained_id[index]]) for index in middle_region_sampling]
  
  # low score region (9/10 ~ 10/10) - bad
  region_start = int(np.maximum(samples_num / 10 * 9, 0))
  region_end = int(np.minimum(samples_num / 10 * 10, samples_num))
  if region_end == 0:
    region_end = 1
  
  low_region_sampling = reorganized_sample_id[region_start: region_end]
  low_region_random_sampling = np.minimum(len(low_region_sampling), random_sampling)
  low_region_sampling = np.random.choice(low_region_sampling, low_region_random_sampling, False)
  low_region_sampling = [_whats_data(data_source,
                                     samples_map[remained_id[index]]['id'],
                                     samples_map[remained_id[index]]) for index in low_region_sampling]

  return reorganized_samples_score, \
         reorganized_model_id, \
         output_reorganized_sample_id, \
         low_region_sampling, \
         middle_region_sampling, \
         high_region_sampling