# -*- coding: UTF-8 -*-
# @Time    : 2019/1/7 10:34 PM
# @File    : evolution.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.automl.graph import *
from antgo.automl.suggestion.models import *
import json
import random
import uuid
from datetime import datetime
from antgo.utils.utils import *
from antgo.automl.suggestion.nsga2 import *
from antgo.utils import logger
import copy
import functools
from antgo.automl.suggestion.mutation import *
from antgo.automl.suggestion.crossover import *
from antgo.automl.suggestion.searchspace.searchspace import *
from itertools import chain
import time


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
    self._dna_func = evolution_s.encoder

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

    mutation_individuals = self.adaptive_mutate(fitness_values=fitness_values,
                                                op_region=self._op_region,
                                                connection_region=(self._block_connection_region_1[0],
                                                                   self._branch_connection_region_2[1]),)

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
            blocks = graph_info['blocks']
            block_i = int(p) // int(self._cell_num*self._branch_num)
            cell_i = (int(p) - block_i * int(self._cell_num*self._branch_num)) // int(self._branch_num)
            branch_i = (int(p) - block_i * int(self._cell_num*self._branch_num)) % int(self._branch_num)
            self._cell.use_channels(graph_info['block_info'][block_i]['base_channel'])
            self._cell.change(graph, cells[blocks[block_i][cell_i]][branch_i])
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
                target = random.choice([0, 1])
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

    self._dna_func = evolution_s.encoder
    self._op_region = (0, self._block_num*self._cell_num*self._branch_num)                      # op type (0~1)
    offset = self._block_num*self._cell_num*self._branch_num

    self._block_connection_region_1 = (offset, offset+1)                                        # 0(no connection), 0.5(random connection), 1(dense connection)
    self._block_connection_region_2 = (offset+1, offset+1+self._block_num*self._block_num)       # 0(no connection), 1(connection)
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
        # change graph
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


class EvolutionSearchSpace(SearchSpace):
  default_params={'max_block_num': 4,
                  'min_block_num': 1,
                  'max_cell_num': 1,
                  'min_cell_num': 1,
                  'branch_num': 5,
                  'base_channel': 64,
                  'channel_mode': 'CONSTANT',
                  'block_stack_mode': 'DECODER',
                  'input_size': '',
                  'population_size': 1,
                  'method': 'nsga2',
                  'max_generation': 100,
                  'mutation_multi_points': 5,
                  'crossover_multi_points': 2,
                  'k0': 0.1,
                  'k1': 1.0,}

  def __init__(self, study, **kwargs):
    super(EvolutionSearchSpace, self).__init__(study, **kwargs)
    self.study = study
    self.population_size = int(kwargs.get('population_size', EvolutionSearchSpace.default_params['population_size']))

    self.current_population = []
    self.current_population_info = []
    if len(Trial.filter(study_name=self.study.name)) == 0:
      # initialize search space (study)
      if self.block_stack_mode == 'DECODER':
        self._initialize_decoder_population()

    self.study_configuration = json.loads(study.study_configuration)
    # build nsga2 evolution algorithm
    mutation_control = EvolutionMutation(self,
                           multi_points=int(kwargs.get('mutation_multi_points',
                                                       EvolutionSearchSpace.default_params['mutation_multi_points'])),
                           generation=self.study_configuration['searchSpace']['current_population_tag'],
                           max_generation=int(kwargs.get('max_generation',
                                                         EvolutionSearchSpace.default_params['max_generation'])),
                           k0=float(kwargs.get('k0',
                                               EvolutionSearchSpace.default_params['k0'])),
                           k1=float(kwargs.get('k1',
                                               EvolutionSearchSpace.default_params['k1'])))

    crossover_control = EvolutionCrossover(self,
                                   multi_points=int(kwargs.get('crossover_multi_points',
                                                               EvolutionSearchSpace.default_params['crossover_multi_points'])),
                                   generation=self.study_configuration['searchSpace']['current_population_tag'],
                                   max_generation=int(kwargs.get('max_generation',
                                                                 EvolutionSearchSpace.default_params['max_generation'])),
                                   k0=float(kwargs.get('k0',
                                                       EvolutionSearchSpace.default_params['k0'])),
                                   k1=float(kwargs.get('k1',
                                                       EvolutionSearchSpace.default_params['k1'])))

    self.evolution_control = Nsga2(ModelProblem('MAXIMIZE'), mutation_control, crossover_control)

  def _initialize_decoder_population(self, random_block_ini=False, random_cell_ini=False):
    # 1.step initialize population
    self.current_population = []
    for _ in range(self.population_size):
      graph_encoder_str, graph_info = self.random(random_block_ini, random_cell_ini)

      self.current_population.append(graph_encoder_str)
      self.current_population_info.append(graph_info)

    # 2.step update study configuration
    study_configuration = json.loads(self.study.study_configuration)
    study_configuration['searchSpace']['current_population'] = self.current_population
    study_configuration['searchSpace']['current_population_info'] = self.current_population_info
    study_configuration['searchSpace']['current_population_tag'] = 0
    self.study.study_configuration = json.dumps(study_configuration)

    # 3.step update study trials
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

  def get_new_suggestions(self, number=1, **kwargs):
    # 1.step get current population tag from study configuration
    study_configuration = json.loads(self.study.study_configuration)
    current_population_tag = int(study_configuration['searchSpace']['current_population_tag'])

    # 2.step fix failed trial (replace those failed trials)
    trials = Trial.filter(study_name=self.study.name, tag=current_population_tag)
    for trial in trials:
      if trial.status == 'Failed':
        logger.warn('trail (id %s) is error and rebuilded'%(trial.name))

        # generate new individual
        graph_encoder_str, graph_info = self.random()
        trial.structure = [graph_encoder_str, graph_info]
        trial.structure_encoder = None
        trial.name = '%s-%s' % (str(uuid.uuid4()), datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))

        temp_graph = Decoder().decode(graph_encoder_str)
        temp_graph.update_by(graph_info)
        trial.multi_objective_value = [1.0 / temp_graph.flops]
        trial.status = None

    # 3.step get candidate trials of study
    candidate_trails = Trial.filter(study_name=self.study.name, tag=current_population_tag, status=None)
    if len(candidate_trails) == 0:
      uncompleted_trails = Trial.filter(study_name=self.study.name, tag=current_population_tag, status="UnCompleted")
      if len(uncompleted_trails) > 0:
        # study not stop, all free worker should wait
        return None

    while len(candidate_trails) == 0:
      # 3.1.step incubate next generation population
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
        grandpa_population = Population()
        grandpa_completed_trials = Trial.filter(study_name=self.study.name, tag=current_population_tag-1, status="Completed")

        for t in grandpa_completed_trials:
          me = self.evolution_control.problem.generateIndividual()
          me.id = t.name
          me.features = [Decoder().decode(t.structure[0]), t.structure[1]]
          me.type = 'parent'
          me.objectives[0] = t.objective_value
          me.objectives[1] = t.multi_objective_value[0]
          self.evolution_control.problem.calculate_objectives(me)
          grandpa_population.population.append(me)

        elite_population = self.evolution_control.evolve(grandpa_population, elite_population)

      # cubate next generation by elite population
      offspring_population = self.evolution_control.create_children(elite_population)
      current_population_tag += 1

      # 3.2.step update trials
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
