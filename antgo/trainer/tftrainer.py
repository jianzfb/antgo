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
import numpy as np
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
  batch_size = getattr(trainer_obj, 'batch_size', 1)
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
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def _get_init_fn(trainer_obj, dump_dir, ctx=None):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  # 1.step load from experiment
  if ctx is not None:
    if ctx.from_experiment is not None:
      # TODO support dencentrid storage in future
      logger.info('load model experiment %s' % ctx.from_experiment.split('/')[-1])
      latest_checkpoint = tf.train.latest_checkpoint(ctx.from_experiment)
      
      variables_to_restore = {}
      for var in slim.get_model_variables():
        var_name = var.op.name
        variables_to_restore[var_name] = var
  
      return [slim.assign_from_checkpoint_fn(latest_checkpoint, variables_to_restore)]

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  # 2.step load from dump_dir
  if tf.train.latest_checkpoint(dump_dir):
    logger.info('Ignoring --checkpoint_path because a checkpoint already exists in %s'% dump_dir)
    # initilize model from dump_dir
    latest_checkpoint = tf.train.latest_checkpoint(dump_dir)
    variables_to_restore = {}
    for var in slim.get_model_variables():
      var_name = var.op.name
      variables_to_restore[var_name] = var

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
  
  transfers = {}
  checkpoint_transfer_scopes = getattr(trainer_obj, 'checkpoint_transfer_scopes', None)
  if checkpoint_transfer_scopes is not None:
    for scope in checkpoint_transfer_scopes.split(','):
      s_scope, t_scope = scope.split(':')
      transfers[s_scope.strip()] = t_scope.strip()
  
  # TODO(sguada) variables.filter_variables()
  auxilary_variables_to_restore = []
  variables_to_restore = {}
  for var in slim.get_model_variables():
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
  
  if trainer_obj.is_training:
    logger.info('fine-tune from %s' % checkpoint_path)
  else:
    logger.info('load from %s' % checkpoint_path)
  return [slim.assign_from_checkpoint_fn(checkpoint_path, vr) for vr in auxilary_variables_to_restore]
  

class TFTrainer(Trainer):
  def __init__(self, trainer_context, dump_dir, is_training=True):
    super(TFTrainer, self).__init__(trainer_context, is_training)
    self.dump_dir = dump_dir

    self.saver = None
    self.sess = None
    self.graph = None
    self.clones = None
    self.lr = None

    self.coord = None
    self.threads = None
  
  # 2.step run model once
  def run(self, data_generator, binds, whats=False):
    # bind data
    feed_dict = {}
    feed_list = []
    with self.graph.as_default():
      for clone in self.clones:
        # generate data
        data = next(data_generator)

        feed_list.append(data)

        for k, v in binds.items():
          placeholder_tensor = self.graph.get_tensor_by_name('{}{}:0'.format(clone.scope, k))
          feed_dict[placeholder_tensor] = data[v]

      # increment
      self.iter_at += 1

      # father method
      super(TFTrainer, self).run(data_generator, binds)
      result = self.sess.run(self.val_ops, feed_dict)

      if self.is_training:
        loss_val = 0.0
        if type(result) == list:
          loss_val = result[1]
        else:
          loss_val = result
        
        if self.iter_at % self.log_every_n_steps == 0:
          logger.info('loss %f lr %f at iterator %d'%(loss_val, self.sess.run(self.lr), self.iter_at))
      return (result, feed_list) if whats else result

  # 3.step snapshot running state
  def snapshot(self, epoch=0):
      assert(self.is_training)
      logger.info('snapshot at %d in %d epoch' % (self.iter_at, epoch))
      if not os.path.exists(self.dump_dir):
          os.makedirs(self.dump_dir)

      model_filename = "{prefix}_{infix}_{d}.ckpt".format(prefix=self.snapshot_prefix,
                                                               infix=self.snapshot_infix, d=self.iter_at)
      model_filepath = os.path.join(self.dump_dir, model_filename)
      self.saver.save(self.sess, model_filepath)

  def training_deploy(self, model):
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      # Session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=graph, config=config)
      
      # initialize data source
      self.ctx.data_source.init(sess=self.sess)
      
      #######################
      # Config model deploy #
      #######################
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=self.num_clones,
                                                      devices=self.devices,
                                                      clone_on_cpu=self.clone_on_cpu,
                                                      replica_id=self.replica_id,
                                                      num_replicas=self.worker_replicas,
                                                      num_ps_tasks=self.num_ps_tasks)

      # Create global_step
      with tf.device(deploy_config.variables_device()):
        global_step = slim.get_or_create_global_step()

      func = model.model_fn
      @functools.wraps(func)
      def network_fn(*args, **kwargs):
        arg_scope = model.arg_scope_fn()
        if arg_scope is not None:
          with slim.arg_scope(arg_scope):
            return func(is_training=self.is_training, *args, **kwargs)
        else:
          return func(is_training=self.is_training, *args, **kwargs)

      #######################
      # Create model clones #
      #######################
      self.clones = tfmodel_deploy.create_clones(deploy_config, network_fn)
      first_clone_scope = deploy_config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

      #########################################
      # Configure the optimization procedure. #
      #########################################
      with tf.device(deploy_config.optimizer_device()):
        num_samples = self.num_samples if self.num_samples > 0 else self.ctx.data_source.size
        self.lr = _configure_learning_rate(self, num_samples, global_step)
        optimizer = _configure_optimizer(self, self.lr)

      # Variables to train.
      variables_to_train = _get_variables_to_train(self)

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

      # Training saver
      self.saver = tf.train.Saver(var_list=slim.get_model_variables(), max_to_keep=2)

      # Global initialization
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())
      
      # Restore from checkpoint
      restore_fns = _get_init_fn(self, self.dump_dir, self.ctx)
      if restore_fns is not None:
        for restore_fn in restore_fns:
          restore_fn(self.sess)

      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

  def infer_deploy(self, model):
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      # Session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=graph, config=config)
      
      # initialize data source
      self.ctx.data_source.init(sess=self.sess)
      
      #######################
      # Config model_deploy #
      #######################
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=1,
                                                      devices=self.devices,
                                                      clone_on_cpu=getattr(self, 'clone_on_cpu', False),
                                                      replica_id=0,
                                                      num_replicas=1,
                                                      num_ps_tasks=0)

      func = model.model_fn
      @functools.wraps(func)
      def network_fn(*args, **kwargs):
        arg_scope = model.arg_scope_fn()
        if arg_scope is not None:
          with slim.arg_scope(arg_scope):
            return func(is_training=self.is_training, *args, **kwargs)
        else:
          return func(is_training=self.is_training, *args, **kwargs)

      #######################
      # Create model clones #
      #######################
      self.clones = tfmodel_deploy.create_clones(deploy_config, network_fn)

      # Restore from checkpoint
      restore_fns = _get_init_fn(self, self.dump_dir, self.ctx)
      if restore_fns is not None:
        for restore_fn in restore_fns:
          restore_fn(self.sess)
      
      # Value ops
      self.val_ops = self.clones[0].outputs
      if type(self.val_ops) != list:
        self.val_ops = [self.val_ops]
        
      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

  # 1.step deploy model on hardwares
  def deploy(self, model):
    # model context
    model.ctx = self._trainer_context

    # deploy model
    if self.is_training:
      self.training_deploy(model)
    else:
      self.infer_deploy(model)
      
  def wait_until_clear(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
