# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:48 PM
# @File    : crossover.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import random


class CrossOver(object):
  def __init__(self, crossover_type, multi_points, adaptive=True, **kwargs):
    self.crossover_type = crossover_type
    self.multi_points = multi_points
    self.adaptive = adaptive

    self.generation = kwargs.get('generation', 0)
    self.max_generation = kwargs.get('max_generation', 1)
    self.k0 = kwargs.get('k0', 0.2)
    self.k1 = kwargs.get('k1', 1.0)

  def _crossover_based_matrices(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    op_regin = kwargs['op_region']              # (start, end)
    block_region = kwargs['block_region']       # (start, end)
    cell_region = kwargs['cell_region']         # (start, end)
    branch_region = kwargs['branch_region']     # (start, end)

    N = len(fitness_values)
    M = fitness_values[0][2].shape[-1]

    ordered_fitness = [(f[0], f[1]) for f in fitness_values]
    ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
    ordered_fitness_values = np.array([m[1] for m in ordered_fitness])
    ordered_fitness_index = [m[0] for m in ordered_fitness]
    confidence = ordered_fitness_values / np.sum(ordered_fitness_values)

    gamma = 1
    if self.adaptive:
      gamma = np.exp(float(self.generation)/float(self.max_generation * self.k0) - self.k1)

    confidence = np.power(confidence, gamma)
    confidence = confidence / np.sum(confidence)

    C = np.zeros((N, 1))      # fitness cumulative probability of chromosome i,
                              # can be considered as an information measure of chromosome i

    c_sum = 0.0
    for a, b in zip(ordered_fitness, confidence):
      index = a[0]
      c_sum += b
      C[index, 0] = c_sum

    A = np.zeros((N, M))
    for n in range(N):
      A[n, :] = fitness_values[n][2]    # gene

    # which position in chromosome i should mutation
    sigma = np.sum(np.power(A - np.mean(A, 0), 2.0) * C, 0) / np.sum(C)
    sigma = sigma / np.sum(sigma)

    information_measure = 1.0 - sigma
    op_information = information_measure[op_regin[0]:op_regin[1]]
    op_information = op_information / (np.sum(op_information) + 0.00000001)
    op_information = np.power(op_information, gamma)
    op_information = op_information / (np.sum(op_information) + 0.00000001)

    connection_information = np.zeros(((block_region[1]-block_region[0]) +
                                       (cell_region[1]-cell_region[0]) +
                                       (branch_region[1]-branch_region[0])))
    connection_information[0:(block_region[1]-block_region[0])] =\
      information_measure[block_region[0]: block_region[1]]

    block_offset = block_region[1]-block_region[0]
    connection_information[block_offset:block_offset+(cell_region[1]-cell_region[0])] = \
      information_measure[cell_region[0]:cell_region[1]]

    cell_offset = block_offset + cell_region[1]-cell_region[0]
    connection_information[cell_offset:cell_offset+(branch_region[1]-branch_region[0])] = \
      information_measure[branch_region[0]:branch_region[1]]

    connection_information = connection_information / (np.sum(connection_information) + 0.000000001)
    connection_information = np.power(connection_information, gamma)
    connection_information = connection_information / (np.sum(connection_information) + 0.000000001)

    crossover_result = []
    for _ in range(N):
      first, second = np.random.choice(ordered_fitness_index, size=2, replace=False, p=confidence)

      # op region cross over
      multi_points_op_region = self.multi_points if self.multi_points > 0 else int(C[first] * op_information.shape[-1])
      op_region_crossover_points = np.random.choice(op_information.shape[-1], multi_points_op_region, replace=False, p=op_information)
      op_region_crossover_points = sorted(op_region_crossover_points)

      # connection region crossover
      multi_points_connection_region = self.multi_points if self.multi_points > 0 else int(C[first]*connection_information.shape[-1])
      connection_region_crossover_points = np.random.choice(connection_information.shape[-1], multi_points_connection_region,replace=False, p=connection_information)
      connection_region_crossover_points = sorted(connection_region_crossover_points)

      ss = []
      for pp in connection_region_crossover_points:
        if pp <block_offset:
          ss.append(pp)
        elif pp>=block_offset and pp <cell_offset:
          ss.append(pp-block_offset+cell_region[0])
        else:
          ss.append(pp-cell_offset+branch_region[0])

      crossovered_first = fitness_values[first] + (second, op_region_crossover_points, ss)
      crossovered_second = fitness_values[second] + (first, op_region_crossover_points, ss)

      crossover_result.append(crossovered_first)
      if len(crossover_result) == N:
        break

      crossover_result.append(crossovered_second)
      if len(crossover_result) == N:
        break

    return crossover_result

  def _crossover_simple(self, *args, **kwargs):
    return None

  def adaptive_crossover(self, *args, **kwargs):
    if self.crossover_type.lower() == 'simple':
      return self._crossover_simple(*args, **kwargs)
    elif self.crossover_type.lower() == 'based_matrices':
      return self._crossover_based_matrices(*args, **kwargs)

    return None

#
# mm = CrossOver('based_matrices', -1, True, generation=10, max_generation=100, k0=0.1, k1=1.0)
#
# fitness_values = []
# for index in range(20):
#   fitness = random.random() * 10
#   gene = np.random.rand((50))
#   rate = random.random()
#   fitness_values.append((index, fitness, gene, rate))
#
# result = mm.adaptive_crossover(fitness_values=fitness_values,op_region=(0,5),block_region=(5,6),cell_region=(12,18),branch_region=(23,30))
