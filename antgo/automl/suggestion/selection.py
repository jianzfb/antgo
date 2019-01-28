# -*- coding: UTF-8 -*-
# @Time    : 2019/1/26 12:47 PM
# @File    : selection.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import random


class Selection(object):
  def __init__(self, selection_type, generation, max_generation, adaptive=True, k0=10):
    self.selection_type = selection_type
    self.generation = generation
    self.max_generation = max_generation
    self.adaptive = adaptive
    self.k0 = k0

  def _tournament_selection(self, fitness_values, select_num, rounds, **kwargs):
    individual_index = [s[0] for s in fitness_values]
    k_individuals = random.sample(individual_index, kwargs['k'])

    for _ in range(rounds):
        best = None
        for index in k_individuals:
          if best is None or kwargs['cmp'](index, best) == 1:
            best = index

        yield best

  def _roulette_wheel_selection(self, fitness_values, select_num, rounds, **kwargs):
    # fitness_values : [(index, fitness),(index, fitness),...]
    sorted_fitness = sorted(fitness_values, key=lambda x: x[1], reverse=True)
    individual_fitness = [s[1] for s in sorted_fitness]
    individual_index = [s[0] for s in sorted_fitness]
    individual_probability = sorted_fitness / np.sum(individual_fitness)

    gamma = 1
    if self.adaptive:
      gamma = np.exp(float(self.generation)/float(self.max_generation * self.k0))

    individual_probability = np.power(individual_probability, gamma)
    individual_probability = individual_probability / np.sum(individual_probability)
    for _ in range(rounds):
      abc = np.random.choice(individual_index, size=select_num, replace=False, p=individual_probability)
      yield abc

  def select(self, fitness_values, select_num, rounds, **kwargs):
    if self.selection_type.lower() == 'tournament':
      self._tournament_selection(fitness_values, select_num, rounds, **kwargs)
    elif self.selection_type.lower() == 'roulette_wheel':
      self._roulette_wheel_selection(fitness_values, select_num, rounds, **kwargs)
