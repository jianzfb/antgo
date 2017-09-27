# -*- coding: UTF-8 -*-
# @Time    : 17-9-19
# @File    : deep_analysis.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np

def discrete_multi_model_measure_analysis(samples_score, data_id, data_source, filter_tag=None, random_sampling=5):
  # 95%, 52%, 42%, 13%, only best, 0%
  # correct is 1; error is 0
  # 95% - sample could be recognized correctly by 95% models
  # 52% - sample could be recognized correctly by 52% models
  # ...
  # filter by tag
  remained_id = []
  if filter_tag is not None:
    for data_index in data_id:
      _, label = data_source.at(data_index)
      if filter_tag in label['tag']:
        remained_id.append(data_index)

  if len(remained_id) == 0:
    remained_id = data_id

  samples_score = samples_score[:, remained_id]
  model_num, samples_num = samples_score.shape[0:2]

  # sort cols
  cols_scores = np.sum(samples_score, axis=0)
  cols_orders = np.argsort(-cols_scores)
  ordered_samples_score = samples_score[:, cols_orders]
  ordered_data_id = [remained_id[i] for i in cols_orders]

  # sort rows
  rows_scores = np.sum(ordered_samples_score, axis=1)
  rows_orders = np.argsort(rows_scores)
  ordered_samples_score = ordered_samples_score[rows_orders, :]
  ordered_model_id = [i for i in rows_orders]

  score_hist = np.sum(ordered_samples_score, axis=0)
  score_hist = score_hist / float(model_num)
  # 95% (94% ~ 96%)
  pos_95 = np.searchsorted(score_hist, [0.95])[0]
  pos_95_s = np.maximum(0, pos_95 - random_sampling/2)
  pos_95_e = np.minimum(samples_num, pos_95_s+random_sampling)
  region_95 = np.arange(pos_95_s, pos_95_e).tolist()

  # 52%
  pos_52 = np.searchsorted(score_hist, [0.52])[0]
  pos_52_s = np.maximum(0, pos_52 - random_sampling/2)
  pos_52_e = np.minimum(samples_num, pos_52_s+random_sampling)
  region_52 = np.arange(pos_52_s, pos_52_e)

  # 42%
  pos_42 = np.searchsorted(score_hist, [0.42])[0]
  pos_42_s = np.maximum(0, pos_42 - random_sampling/2)
  pos_42_e = np.minimum(samples_num, pos_42_s+random_sampling)
  region_42 = np.arange(pos_42_s, pos_42_e)

  # 13%
  pos_13 = np.searchsorted(score_hist, [0.13])[0]
  pos_13_s = np.maximum(0, pos_13 - random_sampling/2)
  pos_13_e = np.minimum(samples_num, pos_13_s+random_sampling)
  region_13 = np.arange(pos_13_s, pos_13_e)

  # only best
  pos_one = np.searchsorted(score_hist, [1.0/float(model_num)])[0]
  pos_one_s = np.maximum(0, pos_one - random_sampling/2)
  pos_one_e = np.minimum(samples_num, pos_one_s+random_sampling)
  region_one = np.arange(pos_one_s,pos_one_e)


  # 0%
  pos_zero = np.searchsorted(score_hist, [0.0])[0]
  pos_zero_s = np.maximum(0, pos_zero - random_sampling / 2)
  pos_zero_e = np.minimum(samples_num, pos_zero_s + random_sampling)
  region_zero = np.arange(pos_zero_s, pos_zero_e)

  return ordered_samples_score, \
         ordered_model_id, \
         ordered_data_id, \
         region_95, \
         region_52, \
         region_42, \
         region_13, \
         region_one, \
         region_zero


def continuous_multi_model_measure_analysis(samples_score, data_id, data_source, filter_tag=None, random_sampling=10):
  # filter by tag
  remained_id = []
  if filter_tag is not None:
    for data_index in data_id:
      _, label = data_source.at(data_index)
      if filter_tag == label['tag']:
        remained_id.append(data_index)
  
  if len(remained_id) == 0:
    remained_id = data_id
  
  samples_score = samples_score[:, remained_id]
  
  # low,  middle,  high
  model_num, samples_num = samples_score.shape[0:2]

  # reorder rows (model)
  model_score = np.sum(samples_score, axis=1)
  reorganized_model_id = np.argsort(-model_score)

  reorganized_samples_score = samples_score[reorganized_model_id.tolist(), :]
  samples_score = reorganized_samples_score.copy()

  # reorder cols (samples)
  s = np.sum(reorganized_samples_score, axis=0)
  reorganized_sample_id = np.argsort(-s)
  reorganized_samples_score = samples_score[:, reorganized_sample_id.tolist()]

  # high score region (0 ~ 1/10) - good
  region_start = 0
  region_end = int(np.minimum(samples_num / 10 * 1, samples_num))
  high_region_sampling = reorganized_sample_id[region_start:region_end]
  high_region_random_sampling = np.minimum(len(high_region_sampling), random_sampling)
  high_region_sampling = np.random.choice(high_region_sampling, high_region_random_sampling, False)
  high_region_sampling = [remained_id[index] for index in high_region_sampling]
  
  # middle score region (4/10 ~ 6/10) - just so so
  region_start = int(np.maximum(samples_num / 10 * 4, 0))
  region_end = int(np.minimum(samples_num / 10 * 6, samples_num))
  middle_region_sampling = reorganized_sample_id[region_start: region_end]
  middle_region_random_sampling = np.minimum(len(middle_region_sampling), random_sampling)
  middle_region_sampling = np.random.choice(middle_region_sampling, middle_region_random_sampling, False)
  middle_region_sampling = [remained_id[index] for index in middle_region_sampling]
  
  # low score region (9/10 ~ 10/10) - bad
  region_start = int(np.maximum(samples_num / 10 * 9, 0))
  region_end = int(np.minimum(samples_num / 10 * 10, samples_num))
  low_region_sampling = reorganized_sample_id[region_start: region_end]
  low_region_random_sampling = np.minimum(len(low_region_sampling), random_sampling)
  low_region_sampling = np.random.choice(low_region_sampling, low_region_random_sampling, False)
  low_region_sampling = [remained_id[index] for index in low_region_sampling]

  # samples score, [], []
  return reorganized_samples_score, \
         reorganized_model_id, \
         reorganized_sample_id, \
         low_region_sampling, \
         middle_region_sampling, \
         high_region_sampling