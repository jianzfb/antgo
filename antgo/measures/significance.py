# encoding=utf-8
# @Time    : 17-8-9
# @File    : significance.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.basic import *
import numpy as np
import random


def bootstrap_confidence_interval(data_source, seed, measure, replicas=50):
  num = data_source.count
  random.seed(seed)
  bootstrap_samples = [[random.randint(0, num-1) for _ in range(num)] for _ in range(replicas)]

  result = measure.eva(data_source.iterate_read('predict', 'groundtruth'), None)
  estimated_measure = result['statistic']['value'][0]['value']

  bootstrap_estimated_measures = []
  for bootstrap_sample in bootstrap_samples:
    record_generator = data_source.iterate_sampling_read(bootstrap_sample, 'predict', 'groundtruth')
    result = measure.eva(record_generator, None)

    value = result['statistic']['value'][0]['value']
    bootstrap_estimated_measures.append(value)

  delta_measure = estimated_measure - np.array(bootstrap_estimated_measures)
  sorted_delta_measure = np.sort(delta_measure)

  # percentile method (B.Efron 1981)
  # confidence interval 95% (alpha = 0.05)
  delta_025 = int(replicas * 0.025)
  delta_975 = int(replicas * 0.975)
  
  return (estimated_measure - sorted_delta_measure[delta_975], estimated_measure - sorted_delta_measure[delta_025])


def bootstrap_direct_confidence_interval(bootstrap_estimated_measures):
  sorted_measures = np.sort(bootstrap_estimated_measures)

  # percentile method (B.Efron 1981)
  # confidence interval 95% (alpha = 0.05)
  pos_025 = int(len(bootstrap_estimated_measures) * 0.025)
  pos_975 = int(len(bootstrap_estimated_measures) * 0.975)

  measure_025 = sorted_measures[pos_025]
  measure_975 = sorted_measures[pos_975]

  return (measure_025, measure_975)


def bootstrap_ab_significance_compare(ab_data_source, seed, measure, replicas=50):
  assert(ab_data_source[0].count == ab_data_source[1].count)
  num = ab_data_source[0].count
  random.seed(seed)
  bootstrap_samples = [[random.randint(0, num-1) for _ in range(num)] for _ in range(replicas)]

  a_bootstrap_scores = []
  for bootstrap_sample in bootstrap_samples:
    record_generator = ab_data_source[0].iterate_sampling_read(bootstrap_sample, 'predict', 'groundtruth')
    result = measure.eva(record_generator, None)
    value = result['statistic']['value'][0]['value']
    a_bootstrap_scores.append(value)

  b_bootstrap_scores = []
  for bootstrap_sample in bootstrap_samples:
    record_generator = ab_data_source[1].iterate_sampling_read(bootstrap_sample, 'predict', 'groundtruth')
    result = measure.eva(record_generator, None)
    value = result['statistic']['value'][0]['value']
    b_bootstrap_scores.append(value)

  diff_bootstrap_scores = np.array(a_bootstrap_scores) - np.array(b_bootstrap_scores)
  sorted_diff_scores = np.sort(diff_bootstrap_scores)

  # percentile method (B.Efron 1981)
  # confidence interval 95% (alpha = 0.05)
  pos_025 = int(replicas * 0.025)
  pos_975 = int(replicas * 0.975)

  if sorted_diff_scores[pos_025] > 0.0:
    if getattr(measure, 'larger', 0) == 1:
      # a is better
      return 1
    else:
      # b is better
      return -1

  if sorted_diff_scores[pos_975] < 0.0:
    if getattr(measure, 'larger', 0) == 1:
      # b is better
      return -1
    else:
      # a is better
      return 1

  # a,b has no significant difference
  return 0

