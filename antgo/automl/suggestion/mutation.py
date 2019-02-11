# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:47 PM
# @File    : mutation.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import random


class Mutation(object):
  def __init__(self, mutation_type, multi_points, adaptive=True, **kwargs):
    self.adaptive = adaptive
    self.mutation_type = mutation_type
    self.multi_points = multi_points    # -1: auto (simulated annealing)
    self.generation = kwargs.get('generation', 0)
    self.max_generation = kwargs.get('max_generation', 1)
    self.k0 = kwargs.get('k0', 0.1)
    self.k1 = kwargs.get('k1', 1.0)

  def _mutate_based_matrices(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    op_region = kwargs['op_region']              # (start, end)
    connection_region = kwargs['connection_region']

    N = len(fitness_values)
    M = fitness_values[0][2].shape[-1]

    C = np.zeros((N, 1))      # fitness cumulative probability of chromosome i,
                              # can be considered as an information measure of chromosome i
    ordered_fitness = [(f[0], f[1]) for f in fitness_values]
    ordered_fitness = sorted(ordered_fitness, key=lambda x: x[1])
    ordered_fitness_values = np.array([m[1] for m in ordered_fitness])
    probability_fitness = ordered_fitness_values / np.sum(ordered_fitness_values)

    gamma = 1
    if self.adaptive:
      gamma = np.exp(float(self.generation)/float(self.max_generation * self.k0) - self.k1)

    probability_fitness = np.power(probability_fitness, gamma)
    probability_fitness = probability_fitness / np.sum(probability_fitness)

    c_sum = 0.0
    for a, b in zip(ordered_fitness, probability_fitness):
      index = a[0]
      c_sum += b
      C[index, 0] = c_sum

    # which individual should mutation
    alpha = 1.0 - C     # the probability to choose which individual for mutation

    A = np.zeros((N, M))

    for n in range(N):
      A[n, :] = fitness_values[n][2]

    # which position in chromosome i should mutation
    sigma = np.sum(np.power(A - np.mean(A, 0), 2.0) * C, 0) / np.sum(C)

    mutation_result = []
    for f in fitness_values:
      # all individual should participate mutation process
      # mutation points number
      multi_points = self.multi_points if self.multi_points > 0 else int(alpha[f[0]] * M)

      # op region mutation
      op_multi_points = np.minimum(multi_points, op_region[1] - op_region[0])
      op_sigma = sigma[op_region[0]:op_region[1]]
      op_sigma = op_sigma / (np.sum(op_sigma) + 0.000000001)
      op_sigma = np.power(op_sigma, gamma)
      op_sigma = op_sigma / (np.sum(op_sigma) + 0.000000001)

      op_mutation_position = np.random.choice(list(range(op_region[1]-op_region[0])),
                                              op_multi_points,
                                              False,
                                              op_sigma)
      op_mutation_position = np.arange(op_region[0],op_region[1])[op_mutation_position].tolist()

      # connection region mutation
      connection_multi_points = np.minimum(multi_points, connection_region[1] - connection_region[0])
      connection_sigma = sigma[connection_region[0]:connection_region[1]]
      connection_sigma = connection_sigma / (np.sum(connection_sigma) + 0.000000001)
      connection_sigma = np.power(connection_sigma, gamma)
      connection_sigma = connection_sigma / (np.sum(connection_sigma) + 0.000000001)

      connection_mutation_position = np.random.choice(list(range(connection_region[1]-connection_region[0])),
                                                      connection_multi_points,
                                                      False,
                                                      connection_sigma)
      connection_mutation_position = np.arange(connection_region[0],connection_region[1])[connection_mutation_position].tolist()
      mutation_result.append(f + (op_mutation_position+connection_mutation_position,))

    return mutation_result

  def _mutate_simple(self, *args, **kwargs):
    # fitness_values: [(index, fitness, gene, rate), (index, fitness, gene, rate), ...]
    fitness_values = kwargs['fitness_values']
    mutation_rate = kwargs['mutation_rate']
    gene_length = fitness_values[0][2].shape[-1]

    mutate_result = []
    for individual in fitness_values:
      if random.random() < mutation_rate:
        mutate_result.append(individual + (None,))
      else:
        mutation_position = random.choice(list(range(gene_length)))
        mutate_result.append((individual + (mutation_position,)))

    return mutate_result

  def adaptive_mutate(self, *args, **kwargs):
    if self.mutation_type.lower() == 'simple':
      return self._mutate_simple(*args, **kwargs)
    elif self.mutation_type.lower() == 'based_matrices':
      return self._mutate_based_matrices(*args, **kwargs)

    return None

# mm = EvMutation('simple', -1, True, generation=10, max_generation=100, k0=0.1, k1=1.0)
#
# fitness_values = []
#
# for index in range(20):
#   fitness = random.random() * 10
#   gene = np.random.rand((10))
#   rate = random.random()
#   fitness_values.append((index, fitness, gene, rate))
#
# result = mm.mutate(fitness_values=fitness_values, mutation_rate=0.5)
