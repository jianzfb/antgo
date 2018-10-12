# -*- coding: UTF-8 -*-
# Time: 8/15/17
# File: tftrainer.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
import os
import re
import functools
from antgo.trainer.trainer import *
from antgo.trainer import tfmodel_deploy
from antgo.utils import logger
from antgo.measures.moving_statistic import *
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from antgo.utils.serialize import *
from antgo.utils.dht import *

import numpy as np
from antgo.utils.net import *
slim = tf.contrib.slim


def _configure_optimizer(trianer_obj, learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if optimizer is not recognized.
  """
  if trianer_obj.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=trianer_obj.adadelta_rho,
        epsilon=trianer_obj.opt_epsilon)
  elif trianer_obj.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=trianer_obj.adagrad_initial_accumulator_value)
  elif trianer_obj.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=trianer_obj.adam_beta1,
        beta2=trianer_obj.adam_beta2,
        epsilon=trianer_obj.opt_epsilon)
  elif trianer_obj.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=trianer_obj.ftrl_learning_rate_power,
        initial_accumulator_value=trianer_obj.ftrl_initial_accumulator_value,
        l1_regularization_strength=trianer_obj.ftrl_l1,
        l2_regularization_strength=trianer_obj.ftrl_l2)
  elif trianer_obj.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=trianer_obj.momentum,
        name='Momentum')
  elif trianer_obj.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=trianer_obj.rmsprop_decay,
        momentum=trianer_obj.rmsprop_momentum,
        epsilon=trianer_obj.opt_epsilon)
  elif trianer_obj.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', trianer_obj.optimizer)
  return optimizer


def _configure_learning_rate(trainer_obj, num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  batch_size = getattr(trainer_obj, 'batch_size', 1) * getattr(trainer_obj, 'num_clones', 1)
  num_epochs_per_decay = getattr(trainer_obj, 'num_epochs_per_decay', 10)
  decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)

  sync_replicas = getattr(trainer_obj, 'sync_replicas', False)
  if sync_replicas:
    replicas_to_aggregate = getattr(trainer_obj, 'replicas_to_aggregate', 1)
    decay_steps /= replicas_to_aggregate

  learning_rate_decay_type = getattr(trainer_obj, 'learning_rate_decay_type', 'exponential')
  learning_rate = getattr(trainer_obj, 'learning_rate', 0.1)
  if learning_rate_decay_type == 'exponential':
    learning_rate_decay_factor = getattr(trainer_obj, 'learning_rate_decay_factor', 0.9)
    return tf.train.exponential_decay(learning_rate,
                                      global_step,
                                      decay_steps,
                                      learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif learning_rate_decay_type == 'fixed':
    return tf.constant(learning_rate, name='fixed_learning_rate')
  elif learning_rate_decay_type == 'polynomial':
    end_learning_rate = getattr(trainer_obj, 'end_learning_rate', 0.0001)
    return tf.train.polynomial_decay(learning_rate,
                                     global_step,
                                     decay_steps,
                                     end_learning_rate,
                                     power=0.9,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized', learning_rate_decay_type)


def _get_variables_to_train(trainer_obj):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  trainable_scopes = getattr(trainer_obj, 'trainable_scopes', None)
  if trainable_scopes is None:
    trainable_filter = getattr(trainer_obj, 'trainable_filter', None)
    if trainable_filter is None:
      return tf.trainable_variables()
    else:
      return [var for var in tf.trainable_variables() if var.name.startswith(trainable_filter)]
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def _get_init_fn(trainer_obj, model, dump_dir, ctx=None):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  # 1.step load from experiment
  if ctx is not None:
    if ctx.from_experiment is not None:
      no_activate_blocks = []
      for b in ctx.blocks():
        if not b.activate:
          no_activate_blocks.append(b.name)

      ablation_folder = '_'.join(sorted(no_activate_blocks))
      
      if len(ablation_folder) > 0:
        ctx.from_experiment = os.path.join(ctx.from_experiment, 'ablation', ablation_folder)
      # print(ctx.from_experiment)
      logger.info('load model from experiment %s' % ctx.from_experiment.split('/')[-2])
      latest_checkpoint = None
      try:
        latest_checkpoint = tf.train.latest_checkpoint(ctx.from_experiment)
      except:
        latest_checkpoint_index = -1
        for f in os.listdir(ctx.from_experiment):
          key_terms = f.split('.')
          if len(key_terms) >= 2 and key_terms[1] == 'ckpt':
            index = int(key_terms[0].split('_')[-1])
            if latest_checkpoint_index < index:
              latest_checkpoint_index = index
              latest_checkpoint = os.path.join(ctx.from_experiment, '.'.join(key_terms[0:2]))
      
      if latest_checkpoint is not None:
        variables_to_restore = {}
        model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables

        exclusions = []
        checkpoint_exclude_scopes = getattr(trainer_obj, 'checkpoint_exclude_scopes', None)
        if checkpoint_exclude_scopes is not None:
          exclusions = [scope.strip()
                        for scope in checkpoint_exclude_scopes.split(',')]
          
          logger.info('exclude scope %s from the latest checkpoint'%(','.join(exclusions)))

        fuzzy_exclusions = []
        checkpoint_fuzzy_exclude_scopes = getattr(trainer_obj, 'checkpoint_fuzzy_exclude_scopes', None)
        if checkpoint_fuzzy_exclude_scopes is not None:
          fuzzy_exclusions = [scope.strip()
                              for scope in checkpoint_fuzzy_exclude_scopes.split(',')]

          logger.info('exclude scope %s from the latest checkpoint' % (','.join(exclusions)))

        for var in model_variables:
          var_name = var.op.name

          excluded = False
          for exclusion in exclusions:
            if var_name.startswith(exclusion):
              excluded = True
              break

          for exclusion in fuzzy_exclusions:
            if exclusion in var_name:
              excluded = True
              break

          if excluded:
            continue
          
          variables_to_restore[var_name] = var
    
        return [slim.assign_from_checkpoint_fn(latest_checkpoint, variables_to_restore, ignore_missing_vars=True)]

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  # 2.step load from dump_dir
  if tf.train.latest_checkpoint(dump_dir):
    logger.info('Ignoring --checkpoint_path because a checkpoint already exists in %s'% dump_dir)
    # initilize model from dump_dir
    latest_checkpoint = tf.train.latest_checkpoint(dump_dir)
    variables_to_restore = {}
    model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables

    for var in model_variables:
      var_name = var.op.name
      variables_to_restore[var_name] = var

    logger.info('load model from %s' % latest_checkpoint)
    return [slim.assign_from_checkpoint_fn(latest_checkpoint, variables_to_restore)]
  
  # 3.step load from checkpoint_path
  checkpoint_path = getattr(trainer_obj, 'checkpoint_path', None)
  if checkpoint_path is None:
    return None

  exclusions = []
  checkpoint_exclude_scopes = getattr(trainer_obj, 'checkpoint_exclude_scopes', None)
  if checkpoint_exclude_scopes is not None:
    exclusions = [scope.strip()
                  for scope in checkpoint_exclude_scopes.split(',')]

  fuzzy_exclusions = []
  checkpoint_fuzzy_exclude_scopes = getattr(trainer_obj, 'checkpoint_fuzzy_exclude_scopes', None)
  if checkpoint_fuzzy_exclude_scopes is not None:
    fuzzy_exclusions = [scope.strip()
                  for scope in checkpoint_fuzzy_exclude_scopes.split(',')]

  transfers = {}
  checkpoint_transfer_scopes = getattr(trainer_obj, 'checkpoint_transfer_scopes', None)
  if checkpoint_transfer_scopes is not None:
    for scope in checkpoint_transfer_scopes.split(','):
      s_scope, t_scope = scope.split(':')
      transfers[s_scope.strip()] = t_scope.strip()
  
  # TODO(sguada) variables.filter_variables()
  auxilary_variables_to_restore = []
  variables_to_restore = {}
  model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
  for var in model_variables:
    # transfer name
    var_name = var.op.name
    for k, v in transfers.items():
      if var.op.name.startswith(k):
        var_name = var_name.replace(k, v)
    
    # exclude name
    excluded = False
    for exclusion in exclusions:
      if var_name.startswith(exclusion):
        excluded = True
        break

    for exclusion in fuzzy_exclusions:
      if exclusion in var_name:
        excluded = True
        break

    # record
    if not excluded:
      if var_name not in variables_to_restore:
        variables_to_restore[var_name] = var
      else:
        is_ok = False
        for auxi_variables in auxilary_variables_to_restore:
          if var_name not in auxi_variables:
            auxi_variables[var_name] = var
            is_ok = True
        
        if not is_ok:
          auxilary_variables_to_restore.append({})
          auxilary_variables_to_restore[-1][var_name] = var
  
  auxilary_variables_to_restore.append(variables_to_restore)
  
  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
  
  logger.info('load model from %s' % checkpoint_path)
  return [slim.assign_from_checkpoint_fn(checkpoint_path, vr, ignore_missing_vars=False) for vr in auxilary_variables_to_restore]


def _convert_to_svg_graph(tf_graph_pb_file, dump_dir, scopes):
  input_graph_def = graph_pb2.GraphDef()
  
  with gfile.FastGFile(tf_graph_pb_file, 'r') as f:
    text_format.Merge(f.read(), input_graph_def)
  
  my_graph = Graph(name='graph')
  # build group table (convolution and batchnorm)
  group_lookup_table = {}
  exclude_lookup_table = {}
  exclude_subfix = {}
  who_is_input = []
  for node in input_graph_def.node:
    terms = node.name.split('/')
    if node.op == 'Placeholder' or terms[0] == 'input':
      who_is_input.append(node.name)

    if terms[0] not in scopes:
      # check convolution group
      if terms[-1] == 'convolution' and node.op == 'Conv2D':
        # special flag
        group_lookup_table['/'.join(terms[0:-1])] = len(terms[0:-1])
      # check batchnorm group
      if len(terms) >= 2:
        if terms[-2] == 'batchnorm' and terms[-1] == 'Rsqrt':
          # special flag
          group_lookup_table['/'.join(terms[0:-2])] = len(terms[0:-2])
      # check AssignmovingAvg group
      if terms[-1].startswith('AssignMovingAvg'):
        group_lookup_table['/'.join(terms[0:-1])] = len(terms[0:-1])
      
      # check dropout group
      if 'dropout' in terms:
        p = 0
        for ti, t in enumerate(terms):
          if t == 'dropout':
            p = ti
        group_lookup_table['/'.join(terms[0:p + 1])] = p + 1
      
      # check optimizer
      if len(terms) >= 3:
        if 'update_discriminator' in terms:
          p = 0
          for ti, t in enumerate(terms):
            if t == 'update_discriminator':
              p = ti
          
          exclude_subfix[terms[p - 1]] = True
          exclude_lookup_table['/'.join(terms[0:p])] = p
      
      # check gradients group
      if len(terms) >= 2:
        if 'gradients' in terms:
          p = 0
          for ti, t in enumerate(terms):
            if t == 'gradients':
              p = ti
          exclude_lookup_table['/'.join(terms[0:p + 1])] = p + 1
      
      # check Relu in convolution scope
      if '/'.join(terms[0:-1]) in group_lookup_table:
        if terms[-1] == 'Relu':
          group_lookup_table['/'.join(terms)] = len(terms)
  
  node_name_map = {}
  for node in input_graph_def.node:
    terms = node.name.split('/')
    name = node.name
    if terms[0] not in scopes:
      # check const op
      if node.op == 'Const':
        continue
      
      # check special node
      if 'Initializer' in terms or 'Assign' in terms or 'read' in terms:
        continue
      
      # check exclude lookup table
      is_exclude = False
      for k, v in exclude_lookup_table.items():
        if name.startswith(k):
          is_exclude = True
          break
      if is_exclude:
        continue
      
      # check exclude subfix
      is_exclude = False
      for k, v in exclude_subfix.items():
        if terms[-1].startswith(k):
          is_exclude = True
          break
      if is_exclude:
        continue
      
      for k, v in group_lookup_table.items():
        if '/'.join(terms[0:v]) == k:
          # check larger match (+1)
          if '/'.join(terms[0:v + 1]) in group_lookup_table:
            name = '/'.join(terms[0:v + 1])
          else:
            name = '/'.join(terms[0:v])
          break
      
      node_name_map[node.name] = name
      # print('%s - %s' % (node.name, name))
      my_graph.add_node(name=name, op_name=name, label='l')
  
  for node in input_graph_def.node:
    terms = node.name.split('/')
    if terms[0] not in scopes:
      if node.op == 'Const':
        continue
      
      if 'Initializer' in terms or 'Assign' in terms or 'read' in terms:
        continue
      
      # check exclude lookup table
      is_exclude = False
      for k, v in exclude_lookup_table.items():
        if node.name.startswith(k):
          is_exclude = True
          break
      if is_exclude:
        continue
      
      # check exclude subfix
      is_exclude = False
      for k, v in exclude_subfix.items():
        if terms[-1].startswith(k):
          is_exclude = True
          break
      if is_exclude:
        continue
      
      input_name = node_name_map[node.name]
      if len(node.input) > 0:
        for linked_node in node.input:
          if linked_node not in node_name_map:
            continue
          
          linked_name = node_name_map[linked_node]
          if linked_name == input_name:
            continue
          
          my_graph.add_link(linked_name, input_name, Link(label='', args=''))
  
  # with open(os.path.join(dump_dir, 'graph.txt'), 'w') as fp:
  #   graph_str = Encoder().encode(my_graph)
  #   fp.write(graph_str)
  
  try:
    svg_graph = graph_net_visualization(my_graph, os.path.join(dump_dir, 'graph.svg'), who_is_input)
    return svg_graph
  except:
    logger.error('dont support graphviz')
    return None


class TFTrainer(Trainer):
  def __init__(self, dump_dir, is_training=True):
    super(TFTrainer, self).__init__(is_training)
    self.dump_dir = dump_dir

    self.saver = None
    self.sess = None
    self.graph = None
    self.clones = None
    self.lr = None

    self.coord = None
    self.threads = None

    self._model_dependence = None

    # record run time
    self.time_stat = MovingAverage(self.log_every_n_steps)
    self.loss_stat = MovingAverage(10)

    self._has_model_input = False
    self.cache = {}

  def run(self, *args, **kwargs):
    data_generator = None

    assert(len(args) <= 1)
    if len(args) != 0:
      data_generator = args[0]

    if data_generator is not None:
      return self._run_by_generator(data_generator, **kwargs)

    return self._run_by_feed(feed_dict=None, **kwargs)

  # 2.step run model once
  def _run_by_generator(self, data_generator, **kwargs):
    # bind data
    with self.graph.as_default():
      feed_dict = {}
      if self._has_model_input and len(self.clones) > 1:
        logger.error('clones number > 1, must set different placeholder for every clone')
        exit(-1)

      if self._has_model_input:
        # generate data
        data = next(data_generator)

        for k, v in kwargs.items():
          if k not in self.cache:
            placeholder_tensor = self.graph.get_tensor_by_name('{}/{}:0'.format('input', k))
            self.cache[k] = placeholder_tensor

          feed_dict[self.cache[k]] = data[v] if (type(data) == tuple or type(data) == list) else data
      else:
        # set different placeholder for every clone
        for clone in self.clones:
          # generate data
          data = next(data_generator)

          for k, v in kwargs.items():
            if k not in self.cache:
              placeholder_tensor = None
              if len(self.clones) > 1:
                placeholder_tensor = self.graph.get_tensor_by_name('{}/{}:0'.format(clone[1][:-1], k))
              else:
                placeholder_tensor = self.graph.get_tensor_by_name('{}:0'.format(k))
              self.cache[k] = placeholder_tensor

            feed_dict[self.cache[k]] = data[v] if (type(data) == tuple or type(data) == list) else data

      return self._run_by_feed(feed_dict=feed_dict, **kwargs)

  def _run_by_feed(self, feed_dict=None, **kwargs):
    with self.graph.as_default():
      if feed_dict is None:
        feed_dict = {}
        for k_name, v_value in kwargs.items():
          if k_name not in self.cache:
            k_tensor = self.graph.get_tensor_by_name('{}:0'.format(k_name))
            self.cache[k_name] = k_tensor

          feed_dict[self.cache[k_name]] = v_value

      # forward process
      start_time = time.time()
      result = self.sess.run(self.val_ops, feed_dict=feed_dict if feed_dict is not None and len(feed_dict) > 0 else None)
      elapsed_time = int((time.time() - start_time) * 100) / 100.0

      if self.ctx.recorder is not None and self.ctx.recorder.model_fn is not None:
        self.ctx.recorder.action(result[-1])
        result = result[:-1]

      self.iter_at += 1

      # record elapsed time
      self.time_stat.add(elapsed_time)

      if self.is_training:
        loss_val = 0.0
        if type(result) == list:
          loss_val = result[1]
        else:
          loss_val = result

        self.loss_stat.add(loss_val)
        if self.iter_at % self.log_every_n_steps == 0:
          logger.info('(PID: %s) INFO: loss %f lr %f at iterator %d (%f sec/step)' %
                      (str(os.getpid()), self.loss_stat.get(), self.sess.run(self.lr), self.iter_at, float(self.time_stat.get())))
      else:
        logger.info('(PID: %s) INFO: (%f sec/step)' % (str(os.getpid()), float(self.time_stat.get())))

      return result[0] if type(result) == list and len(result) == 1 else result

  # 3.step snapshot running state
  def snapshot(self, epoch=0, iter=-1):
    logger.info('snapshot at %d in %d epoch' % (self.iter_at, epoch))
    if not os.path.exists(self.dump_dir):
        os.makedirs(self.dump_dir)

    model_filename = "{prefix}_{infix}_{d}_{e}.ckpt".format(prefix=self.snapshot_prefix,
                                                        infix=self.snapshot_infix,
                                                        d=self.iter_at if iter < 0 else iter,
                                                        e=epoch)
    model_filepath = os.path.join(self.dump_dir, model_filename)
    
    # save checkpoint
    self.saver.save(self.sess, model_filepath)

  def training_deploy(self, model):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      # Session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=graph, config=config)
      
      #######################
      # Config model deploy #
      #######################
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=self.num_clones,
                                                      devices=self.devices,
                                                      clone_on_cpu=self.clone_on_cpu,
                                                      replica_id=self.replica_id,
                                                      num_replicas=self.worker_replicas,
                                                      num_ps_tasks=self.num_ps_tasks)

      # init some info
      with tf.device(deploy_config.inputs_device()):
        #############################
        ####    define model input ##
        #############################
        with tf.variable_scope('input'):
          data_queue = self.ctx.model.model_input(self.is_training)
          if data_queue is not None:
            self._has_model_input = True
        
        #############################
        ####    define model       ##
        #############################
        func = model.model_fn
        @functools.wraps(func)
        def network_fn(*args, **kwargs):
          res = func(self.is_training, *args, **kwargs)
          if kwargs['clone'] == 0:
            # 1.step save graph file
            tf.train.write_graph(self.sess.graph_def, self.dump_dir, 'graph.pbtxt')
            
            # # 2.step transfer to local graph net
            # logger.info('build model graph svg')
            # svg_graph = _convert_to_svg_graph(os.path.join(self.dump_dir, 'graph.pbtxt'),
            #                                   self.dump_dir,
            #                                   ['input'])
            # if svg_graph is not None:
            #   self.ctx.job.send({'DATA': {'GRAPH': svg_graph}})
          return res
  
        #######################
        # Create model clones #
        #######################
        self.clones = tfmodel_deploy.create_clones(deploy_config,
                                                   network_fn,
                                                   [data_queue] if data_queue is not None else None,
                                                   {'trainer': self})
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
          global_step = slim.get_or_create_global_step()

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
          num_samples = self.num_samples if self.num_samples > 0 else self.ctx.data_source.size
          self.lr = _configure_learning_rate(self, num_samples, global_step)
          optimizer = _configure_optimizer(self, self.lr)
  
        # Variables to train.
        variables_to_train = _get_variables_to_train(self)
        
        with tf.control_dependencies(self.model_dependence):
          # Train_tensor
          total_loss, clones_gradients = \
            tfmodel_deploy.optimize_clones(self.clones,
                                           optimizer,
                                           regularization_losses=None if self.regularization_loss else [],
                                           var_list=variables_to_train)
    
          # Create gradient updates.
          grad_updates = optimizer.apply_gradients(clones_gradients,
                                                   global_step=global_step)

        # Value ops
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
          self.val_ops = tf.identity(total_loss, name='train_op')
  
          if self.clones[0].outputs is not None:
            self.val_ops = [self.val_ops]
            if type(self.clones[0].outputs) == list:
              self.val_ops.extend(self.clones[0].outputs)
            elif type(self.clones[0].outputs) == tuple:
              self.val_ops.extend(list(self.clones[0].outputs))
            else:
              self.val_ops.append(self.clones[0].outputs)
        
        # Global initialization
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        # coord
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
        custom_dataset_queue = tf.get_collection('CUSTOM_DATASET_QUEUE')
        if len(custom_dataset_queue) > 0:
          custom_dataset_queue[0].coord = self.coord
          custom_threads = custom_dataset_queue[0].start_threads(self.sess)
          self.threads.extend(custom_threads)
        
        # Training saver
        model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
        self.saver = tf.train.Saver(var_list=model_variables, max_to_keep=2)

        # Restore from checkpoint
        restore_fns = _get_init_fn(self, model, self.dump_dir, self.ctx)
        if restore_fns is not None:
          for restore_fn in restore_fns:
            restore_fn(self.sess)

  def infer_deploy(self, model):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      # Session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=graph, config=config)

      #######################
      # Config model_deploy #
      #######################
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=1,
                                                      devices=self.devices,
                                                      clone_on_cpu=getattr(self, 'clone_on_cpu', False),
                                                      replica_id=0,
                                                      num_replicas=1,
                                                      num_ps_tasks=0)
      
      # init some info
      with tf.device(deploy_config.inputs_device()):
        #############################
        ####    define model input ##
        #############################
        with tf.variable_scope('input'):
          data_queue = self.ctx.model.model_input(self.is_training)
          if data_queue is not None:
            self._has_model_input = True

        #############################
        ####    define model       ##
        #############################
        func = model.model_fn
        @functools.wraps(func)
        def network_fn(*args, **kwargs):
          res = func(self.is_training, *args, **kwargs)
          return res
  
        #######################
        # Create model clones #
        #######################
        self.clones = tfmodel_deploy.create_clones(deploy_config, network_fn, [data_queue])

        # write graph
        tf.train.write_graph(graph.as_graph_def(), self.dump_dir, 'infer_graph.pbtxt')
        # svg_graph = _convert_to_svg_graph(os.path.join(self.dump_dir, 'infer_graph.pbtxt'),
        #                                   self.dump_dir,
        #                                   ['input'])
        # if svg_graph is not None:
        #   self.ctx.job.send({'DATA': {'GRAPH': svg_graph}})

        # Global initialization
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        custom_dataset_queue = tf.get_collection('CUSTOM_DATASET_QUEUE')
        if len(custom_dataset_queue) > 0:
          custom_dataset_queue[0].coord = self.coord
          custom_threads = custom_dataset_queue[0].start_threads(self.sess)
          self.threads.extend(custom_threads)

        # Restore from checkpoint
        restore_fns = _get_init_fn(self, model, self.dump_dir, self.ctx)
        if restore_fns is not None:
          for restore_fn in restore_fns:
            restore_fn(self.sess)
        
        # model saver
        model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
        self.saver = tf.train.Saver(var_list=model_variables, max_to_keep=1)

        # snapshot
        self.snapshot(0)

        # Value ops
        self.val_ops = self.clones[0].outputs
        if type(self.val_ops) != list and type(self.val_ops) != tuple:
          self.val_ops = [self.val_ops]
        if type(self.val_ops) == tuple:
          self.val_ops = list(self.val_ops)

        # Append recorder model fn
        if self.ctx.recorder is not None and self.ctx.recorder.model_fn is not None:
          self.val_ops.append(self.ctx.recorder.model_fn)

  # 1.step deploy model on hardwares
  def deploy(self, model):
    # model context
    self.ctx.model = model
    
    # deploy model
    if self.is_training:
      self.training_deploy(model)
    else:
      self.infer_deploy(model)

  def frozen(self, model):
    assert(not self.is_training)
    self.ctx.model = model
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      self.sess = tf.Session(graph=graph)

      #############################
      ####    define model       ##
      #############################
      model.model_fn(self.is_training)
      # write graph
      tf.train.write_graph(graph.as_graph_def(), self.dump_dir, 'infer_graph.pbtxt')

      # Global initialization
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

      # Restore from checkpoint
      restore_fns = _get_init_fn(self, model, self.dump_dir, self.ctx)
      if restore_fns is not None:
        for restore_fn in restore_fns:
          restore_fn(self.sess)

      # model saver
      model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
      self.saver = tf.train.Saver(var_list=model_variables, max_to_keep=1)

      # snapshot
      self.snapshot(0)

  @property
  def model_dependence(self):
    return self._model_dependence
  @model_dependence.setter
  def model_dependence(self, val):
    self._model_dependence = val
  
  def wait_until_clear(self):
    if self.coord is not None:
      self.coord.request_stop()
      self.coord.join(self.threads)
  