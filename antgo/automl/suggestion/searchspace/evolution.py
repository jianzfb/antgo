# -*- coding: UTF-8 -*-
# @Time    : 2019/1/7 10:34 PM
# @File    : evolution.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.automl.graph import *
from antgo.automl.suggestion.searchspace.abstract_searchspace import *
from antgo.automl.suggestion.models import *
from antgo.automl.suggestion.searchspace.cell import *
from antgo.automl.basestublayers import *
import json
import random
import uuid
from datetime import datetime
from antgo.utils.utils import *
from antgo.automl.suggestion.bayesian import *
from antgo.automl.suggestion.metric import *
from antgo.automl.suggestion.nsga2 import *
from antgo.utils import logger
import copy
import functools
from antgo.automl.suggestion.mutation import *
from antgo.automl.suggestion.crossover import *
from itertools import chain


class EvolutionMutation(Mutation):
  def __init__(self,
               evolution_s,
               multi_points,
               generation,
               max_generation,
               k0,
               k1):
    super(EvolutionMutation, self).__init__('based_matrices',
                                            multi_points,
                                            adaptive=True,
                                            generation=generation,
                                            max_generation=max_generation,
                                            k0=k0,
                                            k1=k1)
    self._block_num = evolution_s.block_num
    self._cell_num = evolution_s.cell_num
    self._branch_num = evolution_s.branch_num

    self._cell = Cell(self._branch_num, evolution_s.base_channel)
    self._dna_func = evolution_s.dna

    self._op_region = (0, self._block_num*self._cell_num*self._branch_num)                      # op type (0~1)
    offset = self._block_num*self._cell_num*self._branch_num

    self._block_connection_region_1 = (offset, offset+1)                                        # 0(no connection), 0.5(random connection), 1(dense connection)
    self._block_connection_region_2 = (offset+1,offset+1+self._block_num*self._block_num)       # 0(no connection), 1(connection)
    self._block_connection_offset = self._op_region[1]
    offset = offset+1+self._block_num*self._block_num

    # 0(no connection), 0.5(random connection), 1(dense connection)
    self._cell_connection_region_1 = (offset, offset+self._block_num)
    # 0(no connection), 1(connection)
    self._cell_connection_region_2 = (offset+self._block_num,
                                      offset+self._block_num+self._block_num*self._cell_num*self._cell_num)
    self._cell_connection_offset = self._block_connection_region_2[1]
    offset = offset+self._block_num+self._block_num*self._cell_num*self._cell_num

    # 0(no connection), 0.5(random connection), 1(dense connection)
    self._branch_connection_region_1 = (offset, offset+self._block_num*self._cell_num)
    # 0(no connection), 1(connection)
    self._branch_connection_region_2 = (offset+self._block_num*self._cell_num,
                                        offset+self._block_num*self._cell_num+
                                        self._block_num*self._cell_num*self._branch_num*self._branch_num)
    self._branch_connection_offset = self._cell_connection_region_2[1]

  def population_mutate(self, *args, **kwargs):
    population = kwargs['population']

    # gene format:  op region:_,_,_,...,_,;
    #               block connection region 1:
    #               block connection region 2: _,_,_,..._,;
    #               cell connection region 1:
    #               cell connection region 2: _,_,_,...,_;
    #               branch connection region 1:
    #               branch connection region 2: _,_,_,...,_;

    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,
                             individual.objectives[0],
                             self._dna_func(individual.features[0], individual.features[1]),
                             None))

    mutation_individuals = self.adaptive_mutate(fitness_values=fitness_values)

    for individual in mutation_individuals:
      if individual[-1] is not None:
        individual_index = individual[0]
        graph, graph_info = population.population[individual_index].features
        mutation_position = individual[-1]
        mutation_position = sorted(mutation_position)

        for p in mutation_position:
          if p >= self._op_region[0] and p < self._op_region[1]:
            # change op as p
            cells = graph_info['cells']
            block_i = int(p) // int(self._cell_num*self._branch_num)
            cell_i = (int(p) - block_i * int(self._cell_num*self._branch_num)) // int(self._branch_num)
            branch_i = (int(p) - block_i * int(self._cell_num*self._branch_num)) % int(self._branch_num)
            self._cell.use_channels(graph_info['block_info'][block_i]['base_channel'])
            new_branch_id = self._cell.change(graph, cells[cell_i][branch_i])
            cells[cell_i][branch_i] = new_branch_id
            graph_info['cells'] = cells

          elif p >= self._block_connection_region_1[0] and p < self._block_connection_region_1[1]:
            # change block connection type
            graph_info['connection']['block'][p-self._block_connection_offset] = random.choice([0, 0.5, 1])
            if graph_info['connection']['block'][p-self._block_connection_offset] == 0:
              graph_info['connection']['block_connection'] = np.zeros((self._block_num*self._block_num)).tolist()
            elif graph_info['connection']['block'][p-self._block_connection_offset] == 1:
              graph_info['connection']['block_connection'] = (np.ones((self._block_num*self._block_num)) - np.eye(self._block_num).flatten()).tolist()
          elif p >= self._block_connection_region_2[0] and p < self._block_connection_region_2[1]:
            if graph_info['connection']['block'][0] != 0 and graph_info['connection']['block'][0] != 1:
              block_i = int(p - self._block_connection_offset - 1) // int(self._block_num)
              block_j = int(p - self._block_connection_offset - 1) % int(self._block_num)
              if block_i != block_j:
                target = random.choice([0, 1])
                graph_info['connection']['block_connection'][block_i * self._block_num + block_j] = target
                graph_info['connection']['block_connection'][block_j * self._block_num + block_i] = target
          elif p >= self._cell_connection_region_1[0] and p < self._cell_connection_region_1[1]:
            # change cell connection type (range size: self._block_num)
            block_i = p - self._cell_connection_offset
            graph_info['connection']['cell'][block_i] = random.choice([0, 0.5, 1])
            if graph_info['connection']['cell'][block_i] == 0:
              graph_info['connection']['cell_connection'][block_i] = np.zeros((self._cell_num*self._cell_num)).tolist()
            elif graph_info['connection']['cell'][block_i] == 1:
              graph_info['connection']['cell_connection'][block_i] = (np.ones((self._cell_num*self._cell_num)) - np.eye(self._cell_num).flatten()).tolist()
          elif p >= self._cell_connection_region_2[0] and p < self._cell_connection_region_2[1]:
            block_i = int(p - self._cell_connection_offset - self._block_num) // int(self._cell_num * self._cell_num)
            if graph_info['connection']['cell'][block_i] != 0 and graph_info['connection']['cell'][block_i] != 1:
              cell_i = (int(p - self._cell_connection_offset - self._block_num) - block_i * int(self._cell_num * self._cell_num)) // int(self._cell_num)
              cell_j = (int(p - self._cell_connection_offset - self._block_num) - block_i * int(self._cell_num * self._cell_num)) % int(self._cell_num)
              if cell_i != cell_j:
                target = random.random([0,1])
                graph_info['connection']['cell_connection'][block_i][cell_i * self._cell_num + cell_j] = target
                graph_info['connection']['cell_connection'][block_i][cell_j * self._cell_num + cell_i] = target
          elif p >= self._branch_connection_region_1[0] and p < self._branch_connection_region_1[1]:
            # change branch connection (range size: self._cell_num * self._block_num)
            cell_i = p - self._branch_connection_offset
            graph_info['connection']['branch'][cell_i] = random.choice([0, 0.5, 1])
            if graph_info['connection']['branch'][cell_i] == 0:
              graph_info['connection']['branch_connection'][cell_i] = np.zeros((self._branch_num*self._branch_num)).tolist()
            elif graph_info['connection']['branch'][cell_i] == 1:
              graph_info['connection']['branch_connection'][cell_i] = (np.ones((self._branch_num*self._branch_num)) - np.eye(self._branch_num).flatten()).tolist()
          elif p >= self._branch_connection_region_2[0] and p < self._branch_connection_region_2[1]:
            cell_i = int(p - self._branch_connection_offset - self._cell_num * self._block_num) // int(self._branch_num * self._branch_num)
            if graph_info['connection']['branch'][cell_i] != 0 and graph_info['connection']['branch'][cell_i] != 1:
              branch_i = int(p - self._branch_connection_offset - self._block_num * self._cell_num - cell_i*self._branch_num*self._branch_num) // int(self._branch_num)
              branch_j = int(p - self._branch_connection_offset - self._block_num * self._cell_num - cell_i*self._branch_num*self._branch_num) % int(self._branch_num)
              if branch_i != branch_j:
                target = random.choice([0, 1])
                graph_info['connection']['branch_connection'][cell_i][branch_i * self._branch_num + branch_j] = target
                graph_info['connection']['branch_connection'][cell_i][branch_j * self._branch_num + branch_i] = target

        population.population[individual_index].features = [graph, graph_info]

    return population


class EvolutionCrossover(CrossOver):
  def __init__(self,
               evolution_s,
               multi_points,
               generation,
               max_generation,
               k0,
               k1):
    super(EvolutionCrossover, self).__init__('based_matrices',
                                             multi_points,
                                             adaptive=True,
                                             generation=generation,
                                             max_generation=max_generation,
                                             k0=k0,
                                             k1=k1)
    self._cell = Cell(evolution_s.branch_num, evolution_s.base_channel)
    self._block_num = evolution_s.block_num
    self._cell_num = evolution_s.cell_num
    self._branch_num = evolution_s.branch_num

    self._dna_func = evolution_s.dna

    self._op_region = (0, self._block_num*self._cell_num*self._branch_num)                      # op type (0~1)
    offset = self._block_num*self._cell_num*self._branch_num

    self._block_connection_region_1 = (offset, offset+1)                                        # 0(no connection), 0.5(random connection), 1(dense connection)
    self._block_connection_region_2 = (offset+1,offset+1+self._block_num*self._block_num)       # 0(no connection), 1(connection)
    self._block_connection_offset = self._op_region[1]
    offset = offset+1+self._block_num*self._block_num

    # 0(no connection), 0.5(random connection), 1(dense connection)
    self._cell_connection_region_1 = (offset, offset+self._block_num)
    # 0(no connection), 1(connection)
    self._cell_connection_region_2 = (offset+self._block_num,
                                      offset+self._block_num+self._block_num*self._cell_num*self._cell_num)
    self._cell_connection_offset = self._block_connection_region_2[1]
    offset = offset+self._block_num+self._block_num*self._cell_num*self._cell_num

    # 0(no connection), 0.5(random connection), 1(dense connection)
    self._branch_connection_region_1 = (offset, offset+self._block_num*self._cell_num)
    # 0(no connection), 1(connection)
    self._branch_connection_region_2 = (offset+self._block_num*self._cell_num,
                                        offset+self._block_num*self._cell_num+
                                        self._block_num*self._cell_num*self._branch_num*self._branch_num)
    self._branch_connection_offset = self._cell_connection_region_2[1]

  def population_crossover(self, *args, **kwargs):
    population = kwargs['population']

    # gene format:  op region:_,_,_,...,_,;
    #               block connection region 1:
    #               block connection region 2: _,_,_,..._,;
    #               cell connection region 1:
    #               cell connection region 2: _,_,_,...,_;
    #               branch connection region 1:
    #               branch connection region 2: _,_,_,...,_;
    fitness_values = []
    for individual_index, individual in enumerate(population.population):
      fitness_values.append((individual_index,
                             individual.objectives[0],
                             self._dna_func(individual.features[0], individual.features[1]),
                             None))

    crossover_individuals = self.adaptive_crossover(fitness_values=fitness_values,
                                                    op_region=self._op_region,
                                                    block_region=self._block_connection_region_1,
                                                    cell_region=self._cell_connection_region_1,
                                                    branch_region=self._branch_connection_region_1)
    crossover_population = Population()
    for individual in crossover_individuals:
      first_index = individual[0]
      second_index = individual[-3]

      crossover_individual = copy.deepcopy(population.population[first_index])
      first_graph, first_graph_info = crossover_individual.features
      second_graph, second_graph_info = population.population[second_index].features

      op_region_crossover_points = individual[-2]
      connection_region_crossover_points = individual[-1]

      for p in op_region_crossover_points:
        first_crossover_b = list(chain(*first_graph_info['cells']))
        first_crossover_b = first_crossover_b[p]

        second_crossover_b = list(chain(*second_graph_info['cells']))
        second_crossover_b = second_crossover_b[p]
        second_crossover_type = second_graph.layer_list[second_crossover_b].layer_name

        target_block = p // (self._cell_num*self._branch_num)
        self._cell.use_channels(first_graph_info['block_info'][target_block]['base_channel'])
        self._cell.change(first_graph, first_crossover_b, second_crossover_type)

      for p in connection_region_crossover_points:
        if p >= self._block_connection_region_1[0] and p < self._block_connection_region_1[1]:
          index = p - self._block_connection_region_1[0]
          first_graph_info['connection']['block'][index] = second_graph_info['connection']['block'][index]
          first_graph_info['connection']['block_connection'] = second_graph_info['connection']['block_connection']
        elif p >= self._cell_connection_region_1[0] and p < self._cell_connection_region_1[1]:
          block_i = p - self._cell_connection_offset
          first_graph_info['connection']['cell'][block_i] = \
            second_graph_info['connection']['cell'][block_i]

          first_graph_info['connection']['cell_connection'][block_i] = \
          second_graph_info['connection']['cell_connection'][block_i]

        elif p >= self._branch_connection_region_1[0] and p < self._branch_connection_region_1[1]:
          cell_i = p - self._branch_connection_offset
          first_graph_info['connection']['branch'][cell_i] = \
            second_graph_info['connection']['branch'][cell_i]

          first_graph_info['connection']['branch_connection'][cell_i] = \
            second_graph_info['connection']['branch_connection'][cell_i]

      crossover_individual.features = [first_graph, first_graph_info]
      crossover_population.population.append(crossover_individual)

    return crossover_population


class ModelProblem(Problem):
  def __init__(self, goal='MAXIMIZE'):
    super(ModelProblem, self).__init__()
    self.max_objectives = [None, None]
    self.min_objectives = [None, None]
    self.goal = goal

  def generateIndividual(self):
    individual = Individual()
    individual.features = []
    individual.dominates = functools.partial(self.__dominates, individual1=individual)
    individual.objectives = [None, None]
    return individual

  def calculate_objectives(self, individual):
    individual.objectives[0] = self.__f1(individual)
    individual.objectives[1] = self.__f2(individual)
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
    # model performance
    return m.objectives[0]

  def __f2(self, m):
    # model flops
    return m.objectives[1]


class EvolutionSearchSpace(AbstractSearchSpace):
  default_params={'max_block_num': 4,
                  'min_block_num': 1,
                  'max_cell_num': 1,
                  'min_cell_num': 1,
                  'branch_num': 5,
                  'base_channel': 64,
                  'channel_mode': 'CONSTANT',
                  'block_stack_mode': 'DECODER',
                  'population_size': 1,
                  'method': 'nsga2',
                  'max_generation': 100,
                  'mutation_multi_points': 5,
                  'crossover_multi_points': 2,
                  'k0': 0.2,
                  'k1': 1.0,
                  'input_size': ''}

  def __init__(self, study, **kwargs):
    super(EvolutionSearchSpace, self).__init__(study, **kwargs)
    self.study = study
    self.study_goal = 'MAXIMIZE'

    self.max_block_num = int(kwargs.get('max_block_num', EvolutionSearchSpace.default_params['max_block_num']))
    self.min_block_num = int(kwargs.get('min_block_num', EvolutionSearchSpace.default_params['min_block_num']))

    self.max_cell_num = int(kwargs.get('max_cell_num', EvolutionSearchSpace.default_params['max_cell_num']))
    self.min_cell_num = int(kwargs.get('min_cell_num', EvolutionSearchSpace.default_params['min_cell_num']))

    self.branch_num = int(kwargs.get('branch_num', EvolutionSearchSpace.default_params['branch_num']))
    self.base_channel = int(kwargs.get('base_channel',
                                       EvolutionSearchSpace.default_params['base_channel']))

    # 'CONSTANT','UP','DOWN'
    self.channel_mode = kwargs.get('channel_mode',
                                   EvolutionSearchSpace.default_params['channel_mode'])

    # ['ENCODER-DECODER', 'ENCODER', 'DECODER']
    self.block_stack_mode = 'DECODER'

    self.input_size = []
    if kwargs.get('input_size') != '':
      input_size = kwargs.get('input_size')
      for s in input_size.split(';'):
        if s == '':
          continue
        a,b,c,d = s.strip().split(',')
        self.input_size.append([int(a),int(b),int(c),int(d)])

    self.block_num = 0
    if len(self.input_size) > 1:
      for index in range(self.max_block_num):
        for input_index in range(len(self.input_size)):
          if index == int(np.log2(self.input_size[input_index][1] / self.input_size[0][1])):
            self.block_num += 1
            break
    else:
      self.block_num = self.max_block_num

    self.cell_num = self.max_cell_num
    self.population_size = int(kwargs.get('population_size', EvolutionSearchSpace.default_params['population_size']))

    self.cell = Cell(branch_num=self.branch_num,
                     base_channel=self.base_channel)

    # temp structure recommand
    # initialize bayesian optimizer
    # self.bayesian = BayesianOptimizer(0.0001, Accuracy, 0.1, 2.576)

    # get all completed trials
    # all_completed_trials = Trial.filter(study_name=self.study.name, status='Completed')
    # x_queue = [np.array(trial.structure_encoder) for trial in all_completed_trials]
    # y_queue = [trial.objective_value for trial in all_completed_trials]
    # if len(x_queue) >= self.population_size:
    #   self.bayesian.fit(x_queue, y_queue)

    self.current_population = []
    self.current_population_info = []
    if len(Trial.filter(study_name=self.study.name)) == 0:
      # initialize search space (study)
      if self.block_stack_mode == 'DECODER':
        self._initialize_decoder_population()

    self.study_configuration = json.loads(study.study_configuration)
    # build nsga2 evolution algorithm
    mutation_control = EvolutionMutation(self,
                           multi_points=int(kwargs.get('mutation_multi_points', EvolutionSearchSpace.default_params['mutation_multi_points'])),
                           generation=self.study_configuration['searchSpace']['current_population_tag'],
                           max_generation=int(kwargs.get('max_generation', EvolutionSearchSpace.default_params['max_generation'])),
                           k0=float(kwargs.get('k0', EvolutionSearchSpace.default_params['k0'])),
                           k1=float(kwargs.get('k1', EvolutionSearchSpace.default_params['k1'])))

    crossover_control = EvolutionCrossover(self,
                                   multi_points=int(kwargs.get('crossover_multi_points', EvolutionSearchSpace.default_params['crossover_multi_points'])),
                                   generation=self.study_configuration['searchSpace']['current_population_tag'],
                                   max_generation=int(kwargs.get('max_generation', EvolutionSearchSpace.default_params['max_generation'])),
                                   k0=float(kwargs.get('k0', EvolutionSearchSpace.default_params['k0'])),
                                   k1=float(kwargs.get('k1', EvolutionSearchSpace.default_params['k1'])))

    self.evolution_control = Nsga2(ModelProblem(self.study_goal), mutation_control, crossover_control)

  def _initialize_decoder_population(self, random_block_ini=False, random_cell_ini=False):
    # 1.step get default graph
    self.current_population = []
    for _ in range(self.population_size):
      self.cell.restore_channels()
      graph_encoder_str, graph_info = self.random(random_block_ini, random_cell_ini)

      self.current_population.append(graph_encoder_str)
      self.current_population_info.append(graph_info)

    # write study table
    study_configuration = json.loads(self.study.study_configuration)
    study_configuration['searchSpace']['current_population'] = self.current_population
    study_configuration['searchSpace']['current_population_info'] = self.current_population_info
    study_configuration['searchSpace']['current_population_tag'] = 0
    self.study.study_configuration = json.dumps(study_configuration)

    # write trials table
    for graph_str, graph_info in zip(self.current_population, self.current_population_info):
      trail_name = '%s-%s' % (str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))
      trial = Trial.create(Trial(self.study.name, trail_name, created_time=time.time(), updated_time=time.time()))
      trial.structure = [graph_str, graph_info]
      trial.structure_encoder = None
      trial.objective_value = -1

      temp_graph = Decoder().decode(graph_str)
      temp_graph.update_by(graph_info)
      trial.multi_objective_value = [1.0 / temp_graph.flops]
      trial.tag = 0     # 0 generation

  def dna(self, graph, graph_info):
    # graph only include no skip connection
    block_num = len(graph_info['blocks'])
    op_length = block_num * self.max_cell_num * self.branch_num
    connection_length = 1 + block_num * block_num + \
                        block_num + block_num * self.max_cell_num * self.max_cell_num + \
                        block_num * self.max_cell_num + \
                        block_num * self.max_cell_num * self.branch_num * self.branch_num

    dna_vector = np.zeros((op_length+connection_length))

    blocks = graph_info['blocks']
    cells = graph_info['cells']
    # for op region
    for block_index, block in enumerate(blocks):
      dna_block_offset = block_index * self.max_cell_num * self.branch_num

      for cell_index, cell_id in enumerate(block):
        dna_cell_offset = cell_index * self.branch_num

        for branch_index, branch_id in enumerate(cells[cell_id]):
          dna_vector[dna_block_offset + dna_cell_offset + branch_index] =\
            graph.layer_list[branch_id].layer_type_encoder

    # for connection region
    # 1.step block connection region
    offset = op_length
    dna_vector[offset] = graph_info['connection']['block'][0]
    dna_vector[offset+1:offset+1+block_num*block_num] = np.array(graph_info['connection']['block_connection'])

    # 2.step cell connection region
    offset += 1 + block_num * block_num
    dna_vector[offset:offset+block_num] = np.array(graph_info['connection']['cell'])

    long_v = []
    for bb in range(block_num):
      long_v += graph_info['connection']['cell_connection'][bb]

    dna_vector[offset+block_num:
               offset+block_num+block_num * self.max_cell_num * self.max_cell_num] = np.array(long_v)


    # 3.step branch connection region
    offset += block_num + block_num * self.max_cell_num * self.max_cell_num
    dna_vector[offset:offset+block_num * self.max_cell_num] = np.array(graph_info['connection']['branch'])
    offset_in = offset + block_num * self.max_cell_num

    long_v = []
    for cc in range(len(cells)):
      long_v += graph_info['connection']['branch_connection'][cc]

    dna_vector[offset_in:] = np.array(long_v)
    return dna_vector

  def random(self, random_block_ini=False, random_cell_ini=False, mode='decoder'):
    # 1.step get block number
    blocks = [[]]
    block_info = {0: {'mode': 'u', 'size': 0, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}}
    for index in range(self.max_block_num):
      if index == 0:
        continue

      if random.random() < 0.5 and random_block_ini:
        block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
        blocks.append([])
      else:
        if len(self.input_size) > 1:
          for input_index in range(len(self.input_size)):
            if index == int(np.log2(self.input_size[input_index][1] / self.input_size[0][1])):
              block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
              blocks.append([])
              break
        else:
          block_info[len(blocks)] = {'mode': 'u', 'size': index, 'transition': -1, 'input_transition': [], 'base_channel': self.base_channel}
          blocks.append([])

    block_num = len(blocks)

    # 2.step get cell number in every block
    cell_num = []
    for _ in range(block_num):
      if random_cell_ini:
        cell_num.append(random.choice(list(range(self.min_cell_num, self.max_cell_num))))
      else:
        cell_num.append(self.max_cell_num)

    # 3.step generate cells and branches
    default_graph = Graph()
    default_graph.layer_factory = BaseLayerFactory()
    for input_s in self.input_size:
      default_graph.add_input(shape=input_s)

    last_layer_id = -1

    cells = []
    cell_id_offset = 0
    last_channel = self.cell.current_channel
    for block_id in range(block_num):
      if block_id == 0:
        for cell_id in range(cell_num[block_id]):
          cells.append(self.cell.random(default_graph,
                                        last_layer_id,
                                        cell_name='cell_%d' % (cell_id + cell_id_offset),
                                        block_name='block_%d' % block_info[block_id]['size']))

          last_layer_id = cells[-1][-1]
          blocks[block_id].append(cell_id + cell_id_offset)
      else:
        # adjust block channels
        if self.channel_mode == 'UP':
          self.cell.increase_channels(2.0)
        elif self.channel_mode == 'DOWN':
          self.cell.decrease_channels(0.5)

        block_info[block_id]['base_channel'] = self.cell.current_channel

        # 1.step channel compatibility
        if self.cell.current_channel != last_channel:
          # 3x3 convolution layer
          transition_conv33 = default_graph.layer_factory.conv2d(input_channel=None,
                                                                 filters=self.cell.current_channel,
                                                                 kernel_size_h=3,
                                                                 kernel_size_w=3)

          default_graph.add_layer(transition_conv33, default_graph.layer_id_to_output_node_ids[last_layer_id][0])
          last_layer_id = default_graph.layer_to_id[transition_conv33]
          last_channel = self.cell.current_channel

        # 2.step spatial compatibility
        base_size = int(np.power(2, block_info[block_id]['size']))
        block_transition_layer = default_graph.layer_factory.bilinear_resize(height=self.input_size[0][1] * base_size,
                                                                             width=self.input_size[0][2] * base_size)
        default_graph.add_layer(block_transition_layer, default_graph.layer_id_to_output_node_ids[last_layer_id][0])
        last_layer_id = default_graph.layer_to_id[block_transition_layer]

        block_info[block_id]['transition'] = last_layer_id

        for cell_id in range(cell_num[block_id]):
          cells.append(self.cell.random(default_graph,
                                        last_layer_id,
                                        cell_name='cell_%d' % (cell_id + cell_id_offset),
                                        block_name='block_%d' % block_info[block_id]['size']))

          last_layer_id = cells[-1][-1]
          blocks[block_id].append(cell_id + cell_id_offset)

        # 3.step link with compatible input
        for input_i, input_s in enumerate(self.input_size):
          if input_s[1] == self.input_size[0][1] * base_size and input_s[2] == self.input_size[0][2] * base_size:
            # identity layer
            identity_layer = default_graph.layer_factory.identity()
            default_graph.add_layer(identity_layer, default_graph.get_input()[input_i])

            # add skip model
            default_graph.to_add_skip_model(default_graph.layer_to_id[identity_layer],
                                            block_info[block_id]['transition'])
            default_graph.update()

      cell_id_offset += cell_num[block_id]

    # 4.step build connection information between blocks,cells, branches
    block_connection_type = random.choice([0,0.5,1])
    block_connections = []
    if block_connection_type == 0:
      block_connections = np.zeros((block_num * block_num)).tolist()
    elif block_connection_type == 1:
      block_connections = (np.ones((block_num * block_num)) - np.eye(block_num).flatten()).tolist()
    elif block_connection_type == 0.5:
      block_connections = np.zeros((block_num * block_num)).tolist()
      for bi in range(block_num):
        for bj in range(block_num):
          if bj > bi:
            target = random.choice([0,1])
            block_connections[bi * block_num + bj] = target
            block_connections[bj * block_num + bi] = target

    cell_connection_type = np.random.choice([0, 0.5, 1], block_num).tolist()
    cell_connections = [np.zeros(self.max_cell_num * self.max_cell_num).tolist() for _ in range(block_num)]
    for b in range(block_num):
      if cell_connection_type[b] == 0:
        cell_connections[b] = np.zeros(self.max_cell_num * self.max_cell_num).tolist()
      elif cell_connection_type[b] == 1:
        cell_connections[b] = (np.ones(self.max_cell_num * self.max_cell_num) - np.eye(self.max_cell_num).flatten()).tolist()
      else:
        cell_connections[b] = np.zeros(self.max_cell_num * self.max_cell_num).tolist()
        for ci in range(self.max_cell_num):
          for cj in range(self.max_cell_num):
            if cj > ci:
              target = random.choice([0, 1])
              cell_connections[b][ci * self.max_cell_num + cj] = target
              cell_connections[b][cj * self.max_cell_num + ci] = target

    branch_connection_type = np.random.choice([0, 0.5, 1], block_num * self.max_cell_num).tolist()
    branch_connections = [np.zeros(self.branch_num * self.branch_num).tolist() for _ in
                          range(block_num * self.max_cell_num)]
    for c in range(block_num * self.max_cell_num):
      if branch_connection_type[c] == 0:
        branch_connections[c] = np.zeros(self.branch_num * self.branch_num).tolist()
      elif branch_connection_type[c] == 1:
        branch_connections[c] = (np.ones(self.branch_num * self.branch_num)-np.eye(self.branch_num).flatten()).tolist()
      else:
        branch_connections[c] = np.zeros(self.branch_num * self.branch_num).tolist()
        for bi in range(self.branch_num):
          for bj in range(self.branch_num):
            if bj > bi:
              target = random.choice([0, 1])
              branch_connections[c][bi * self.branch_num + bj] = target
              branch_connections[c][bj * self.branch_num + bi] = target

    default_graph.layer_factory = None
    graph_encoder_str = Encoder(skipkeys=True).encode(default_graph)
    graph_info = {'blocks': blocks,
                  'cells': cells,
                  'block_info': block_info,
                  'connection': {'block': [block_connection_type],
                                 'block_connection': block_connections,
                                 'cell': cell_connection_type,
                                 'cell_connection': cell_connections,
                                 'branch': branch_connection_type,
                                 'branch_connection': branch_connections}}

    return graph_encoder_str, graph_info

  def get_new_suggestions(self, number=1, **kwargs):
    # get current population
    study_configuration = json.loads(self.study.study_configuration)
    current_population_tag = int(study_configuration['searchSpace']['current_population_tag'])

    # update those failed trails (status=='Failed' or expired)
    trials = Trial.filter(study_name=self.study.name, tag=current_population_tag)
    for trial in trials:
      if trial.status == 'Failed' or \
              ((time.time() - float(trial.updated_time)) >= 2 * 60 * 60 and trial.status == 'UnCompleted'):
        #
        error_reason = 'running error' if trial.status == 'Failed' else 'expired'
        logger.warn('trail (id%s) is error and rebuilded (reason: %s)'%(trial.name, error_reason))

        # generate new individual
        graph_encoder_str, graph_info = self.random()
        trial.structure = [graph_encoder_str, graph_info]
        trial.structure_encoder = None

        temp_graph = Decoder().decode(graph_encoder_str)
        temp_graph.update_by(graph_info)
        trial.multi_objective_value = [1.0 / temp_graph.flops]
        trial.status = None

    candidate_trails = Trial.filter(study_name=self.study.name, tag=current_population_tag, status=None)

    if len(candidate_trails) == 0:
      uncompleted_trails = Trial.filter(study_name=self.study.name, tag=current_population_tag, status="UnCompleted")
      if len(uncompleted_trails) > 0:
        # study not stop, all free worker should wait
        return None

    while len(candidate_trails) == 0:
      # get elite population
      elite_population = Population()
      completed_trials = Trial.filter(study_name=self.study.name, tag=current_population_tag, status="Completed")

      for t in completed_trials:
        me = self.evolution_control.problem.generateIndividual()
        me.id = t.name
        me.features = [Decoder().decode(t.structure[0]), t.structure[1]]
        me.type = 'parent'
        me.objectives[0] = t.objective_value
        me.objectives[1] = t.multi_objective_value[0]
        self.evolution_control.problem.calculate_objectives(me)
        elite_population.population.append(me)

      if current_population_tag >= 1:
        parent_population = Population()
        parent_completed_trials = Trial.filter(study_name=self.study.name, tag=current_population_tag-1, status="Completed")

        for t in parent_completed_trials:
          me = self.evolution_control.problem.generateIndividual()
          me.id = t.name
          me.features = [Decoder().decode(t.structure[0]), t.structure[1]]
          me.type = 'parent'
          me.objectives[0] = t.objective_value
          me.objectives[1] = t.multi_objective_value[0]
          self.evolution_control.problem.calculate_objectives(me)
          parent_population.population.append(me)

        elite_population = self.evolution_control.evolve(parent_population, elite_population)

      offspring_population = self.evolution_control.create_children(elite_population)
      current_population_tag += 1

      # generate trials
      study_current_population = []
      study_current_population_info = []
      for p in offspring_population.population:
        trail_name = '%s-%s' % (str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))
        trial = Trial.create(Trial(self.study.name, trail_name, created_time=time.time(), updated_time=time.time()))
        trial.structure = [Encoder(skipkeys=True).encode(p.features[0]), p.features[1]]
        trial.structure_encoder = None

        temp_graph = Decoder().decode(trial.structure[0])
        temp_graph.update_by(trial.structure[1])
        trial.objective_value = -1
        trial.multi_objective_value = [1.0 / temp_graph.flops]
        trial.tag = current_population_tag

        study_current_population.append(trial.structure[0])
        study_current_population_info.append(trial.structure[1])

      # update study configuration
      study_configuration['searchSpace']['current_population'] = study_current_population
      study_configuration['searchSpace']['current_population_info'] = study_current_population_info
      study_configuration['searchSpace']['current_population_tag'] = current_population_tag

      # regenerate candidate trials
      candidate_trails = Trial.filter(study_name=self.study.name, tag=current_population_tag, status=None)

    self.study.study_configuration = json.dumps(study_configuration)
    trial_suggestion = random.choice(candidate_trails)
    trial_suggestion.status = 'UnCompleted'
    return [trial_suggestion]
