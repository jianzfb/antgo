# -*- coding: UTF-8 -*-
# @Time    : 2019/1/16 7:11 PM
# @File    : nsga2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import math
import random
import functools
import copy
# import matplotlib.pyplot as plt


class Population(object):
  """Represents population - a group of Individuals,
  can merge with another population"""

  def __init__(self):
    self.population = []
    self.fronts = []

  def __len__(self):
    return len(self.population)

  def __iter__(self):
    """Allows for iterating over Individuals"""

    return self.population.__iter__()

  def extend(self, new_individuals):
    """Creates new population that consists of
    old individuals ans new_individuals"""

    self.population.extend(new_individuals)


class Individual(object):
  """Represents one individual"""

  def __init__(self):
    self.rank = None
    self.crowding_distance = None
    self.dominated_solutions = set()
    self.features = None
    self.objectives = None
    self.dominates = None
    self.id = None
    self.type = 'parent'

  def set_objectives(self, objectives):
    self.objectives = objectives


class Problem(object):
  def __init__(self):
    self.max_objectives = None
    self.min_objecives = None
    self.problem_type = None

  def calculate_objectives(self, individual):
    raise NotImplementedError


class Nsga2(object):
  def __init__(self, problem, mutation_op, crossover_op):
    self.mutation_op = mutation_op
    self.crossover_op = crossover_op

    self.solution = []
    self.multi_objects = []
    self.problem = problem

  def fast_nondominated_sort(self, population):
    population.fronts = []
    population.fronts.append([])

    # clear
    for individual in population:
      individual.domination_count = 0
      individual.dominated_solutions = set()
      individual.rank = None

    # make statistic
    for individual in population:
      individual.domination_count = 0
      individual.dominated_solutions = set()

      for other_individual in population:
        if individual.dominates(other_individual):
          individual.dominated_solutions.add(other_individual)
        elif other_individual.dominates(individual):
          individual.domination_count += 1

      if individual.domination_count == 0:
        population.fronts[0].append(individual)
        individual.rank = 0
    i = 0
    while len(population.fronts[i]) > 0:
      temp = []
      for individual in population.fronts[i]:
        for other_individual in individual.dominated_solutions:
          other_individual.domination_count -= 1
          if other_individual.domination_count == 0:
            other_individual.rank = i + 1
            temp.append(other_individual)
      i = i + 1
      population.fronts.append(temp)

  def crowding_operator(self, individual, other_individual):
    if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (
                    individual.crowding_distance > other_individual.crowding_distance)):
      return 1
    else:
      return -1

  def calculate_crowding_distance(self, front):
    if len(front) > 0:
      solutions_num = len(front)
      for individual in front:
        individual.crowding_distance = 0

      for m in range(len(front[0].objectives)):
        front = sorted(front, key=lambda x: x.objectives[m])

        # front[0].crowding_distance = self.problem.max_objectives[m]
        # front[solutions_num - 1].crowding_distance += self.problem.max_objectives[m]

        front[0].crowding_distance = 4444444444444
        front[solutions_num - 1].crowding_distance = 4444444444

        for index, value in enumerate(front[1:solutions_num - 1]):
          front[index].crowding_distance += (front[index + 1].objectives[m] - front[index - 1].objectives[m]) / (
                    self.problem.max_objectives[m] - self.problem.min_objectives[m])

  def __tournament(self, population):
    participants = random.sample(population.population, 2)
    best = None
    for participant in participants:
      if best is None or self.crowding_operator(participant, best) == 1:
        best = participant

    return best

  def create_children(self, population):
    children = []
    pop_size = len(population)
    while len(children) != pop_size:
      best = self.__tournament(population)

      structure = copy.deepcopy(best.features[0])
      structure_info = copy.deepcopy(best.features[1])
      structure, structure_info = self.mutation_op.mutate(structure, structure_info)

      me = self.problem.generateIndividual()
      me.features = [structure, structure_info]
      me.type = 'offspring'

      self.problem.calculate_objectives(me)
      children.append(me)

    return children

  def evolve(self, population):
    population_size = len(population)
    # 1.step compute nondominated_sort and crowding distance
    self.fast_nondominated_sort(population)
    for front in population.fronts:
      self.calculate_crowding_distance(front)

    # 2.step generate next children generation
    children = self.create_children(population)

    # 3.step environment pooling
    # 3.1.step expand population
    population.extend(children)

    # 3.2.step re-fast-nondominated-sort
    self.fast_nondominated_sort(population)

    # 3.3.step build new population
    new_population = Population()
    front_num = 0
    while len(new_population) + len(population.fronts[front_num]) <= population_size:
      new_population.extend(population.fronts[front_num])
      front_num += 1

    if len(new_population) < population_size:
      self.calculate_crowding_distance(population.fronts[front_num])
      population.fronts[front_num] = sorted(population.fronts[front_num],
                                                 key=functools.cmp_to_key(self.crowding_operator),
                                            reverse=True)
      new_population.extend(population.fronts[front_num][0:population_size - len(new_population)])
    return new_population


# test nsga2
class ZDT1(Problem):
  def __init__(self, goal='MAXIMIZE'):
    super(ZDT1, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal

  def generateIndividual(self):
    individual = Individual()
    individual.features = []

    min_x = -55
    max_x = 55

    individual.features.append(min_x + (max_x - min_x) * random.random())
    individual.features.append(None)
    individual.dominates = functools.partial(self.__dominates, individual1=individual)
    return individual

  def calculate_objectives(self, individual):
    individual.objectives = []
    individual.objectives.append(self.__f1(individual))
    individual.objectives.append(self.__f2(individual))
    for i in range(2):
      if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
        self.min_objectives[i] = individual.objectives[i]
      if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
        self.max_objectives[i] = individual.objectives[i]

  def __dominates(self, individual2, individual1):
    if self.goal == 'MAXIMIZE':
      worse_than_other = self.__f1(individual1) >= self.__f1(individual2) and self.__f2(individual1) >= self.__f2(
        individual2)
      better_than_other = self.__f1(individual1) > self.__f1(individual2) or self.__f2(individual1) > self.__f2(
        individual2)
      return worse_than_other and better_than_other
    else:
      worse_than_other = self.__f1(individual1) <= self.__f1(individual2) and self.__f2(individual1) <= self.__f2(
        individual2)
      better_than_other = self.__f1(individual1) < self.__f1(individual2) or self.__f2(individual1) < self.__f2(
        individual2)
      return worse_than_other and better_than_other

  def __f1(self, m):
    value = (m.features[0]**2)
    return value

  def __f2(self, m):
    value = (m.features[0]-2)**2
    return value


if __name__ == '__main__':
  class _ZDT1Mutation(object):
    def __init__(self):
      pass

    def mutate(self, a, b):
      min_x = -55
      max_x = 55
      v = min_x+(max_x-min_x)*random.random()
      return v, b

  problem = ZDT1('MINIMIZE')
  ss = Nsga2(problem, _ZDT1Mutation(), None)

  population = Population()
  num_of_individuals = 20
  for _ in range(num_of_individuals):
    individual = problem.generateIndividual()
    problem.calculate_objectives(individual)
    population.population.append(individual)

  new_population = population
  for index in range(46):
    new_population = ss.evolve(new_population)

    function1_values = [m.objectives[0] for m in new_population.population]
    function2_values = [m.objectives[1] for m in new_population.population]
    function1 = [i  for i in function1_values]
    function2 = [j  for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2, c='r')
    plt.show()