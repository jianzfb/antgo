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


def _configure_optimizer(optimization_config, learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if optimizer is not recognized.
  """
  if optimization_config['optimizer'] == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=optimization_config['adadelta_rho'],
        epsilon=optimization_config['opt_epsilon'])
  elif optimization_config['optimizer'] == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=optimization_config['adagrad_initial_accumulator_value'])
  elif optimization_config['optimizer'] == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=optimization_config['adam_beta1'],
        beta2=optimization_config['adam_beta2'],
        epsilon=optimization_config['opt_epsilon'])
  elif optimization_config['optimizer'] == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=optimization_config['ftrl_learning_rate_power'],
        initial_accumulator_value=optimization_config['ftrl_initial_accumulator_value'],
        l1_regularization_strength=optimization_config['ftrl_l1'],
        l2_regularization_strength=optimization_config['ftrl_l2'])
  elif optimization_config['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=optimization_config['momentum'],
        name='Momentum')
  elif optimization_config['optimizer'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=optimization_config['rmsprop_decay'],
        momentum=optimization_config['rmsprop_momentum'],
        epsilon=optimization_config['opt_epsilon'])
  elif optimization_config['optimizer'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimization_config['optimizer'])
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

  sync_replicas = getattr(trainer_obj, 'sync_replicas', None)
  if sync_replicas is not None:
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


def _get_init_fn(trainer_obj, dump_dir):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  checkpoint_path = getattr(trainer_obj, 'checkpoint_path', None)
  if checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(dump_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'% dump_dir)
    return None

  exclusions = []
  checkpoint_exclude_scopes = getattr(trainer_obj, 'checkpoint_exclude_scopes', None)
  if checkpoint_exclude_scopes is not None:
    exclusions = [scope.strip()
                  for scope in checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    print(var.name)

    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)
  return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


class TFTrainer(Trainer):
  def __init__(self, trainer_context, dump_dir, is_training=True):
    super(TFTrainer, self).__init__(trainer_context, is_training)
    self.dump_dir = dump_dir

    self.saver = None
    self.sess = None
    self.graph = None
    self.clones = None

  # 2.step run model once
  def run(self, data_generator, binds):
    # bind data
    feed_dict = {}
    with self.graph.as_default():
      for clone in self.clones:
        # generate data
        data = next(data_generator)

        for k, v in binds.items():
          placeholder_tensor = self.graph.get_tensor_by_name('{}/{}:0'.format(clone.scope.name, k))
          feed_dict[placeholder_tensor] = data[v]

      # modify running info
      self.iter_at += 1

      # father method
      super(TFTrainer, self).run(data_generator, binds)
      return self.sess.run(self.val_ops, feed_dict)

  # 3.step snapshot running state
  def snapshot(self, epoch=0):
      assert(self.is_training)
      logger.info('snapshot at %d in %d epoch' % (self.iter_at, epoch))
      if not os.path.exists(self.dump_dir):
          os.makedirs(self.dump_dir)

      model_filename = "{prefix}_{infix}_iter_{d}.ckpt".format(prefix=self.snapshot_prefix,
                                                               infix=self.snapshot_infix, d=self.iter_at)
      model_filepath = os.path.join(self.dump_dir, model_filename)
      self.saver.save(self.sess, model_filepath)

  def where_latest_model(self):
      latest_index = -1
      pattern_str = "(?<={prefix}_{infix}_iter_)\d+(?=.ckpt)".format(prefix=self.snapshot_prefix,
                                                                     infix=self.snapshot_infix)

      if os.path.exists(self.dump_dir):
          for file in os.listdir(self.dump_dir):
              match_result = re.findall(pattern_str, file)
              if len(match_result) > 0:
                  if int(match_result[0]) > latest_index:
                      latest_index = int(match_result[0])

          if latest_index > 0:
              self.iter_at = latest_index + 1
              return os.path.join(self.dump_dir,
                                  "{prefix}_{infix}_iter_{d}.ckpt".format(prefix=self.snapshot_prefix,
                                                                          infix=self.snapshot_infix, d=latest_index))

      return None

  def training_deploy(self, model):
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
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=getattr(self, 'num_clones', 1),
                                                      clone_on_cpu=getattr(self, 'clone_on_cpu', False),
                                                      replica_id=getattr(self, 'replica_id', 0),
                                                      num_replicas=getattr(self, 'worker_replicas', 1),
                                                      num_ps_tasks=getattr(self, 'num_ps_tasks', 0))

      # Create global_step
      with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

      func = model.build
      @functools.wraps(func)
      def network_fn(*args, **kwargs):
        arg_scope = model.arg_scope()
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
        num_samples = getattr(self, 'num_samples', 10000)
        learning_rate = _configure_learning_rate(self, num_samples, global_step)
        optimizer = _configure_optimizer(self.optimization, learning_rate)

      # Variables to train.
      variables_to_train = _get_variables_to_train(self)

      # Train_tensor
      total_loss, clones_gradients = tfmodel_deploy.optimize_clones(
        self.clones,
        optimizer,
        var_list=variables_to_train)

      # Create gradient updates.
      grad_updates = optimizer.apply_gradients(clones_gradients,
        global_step=global_step)

      # Value ops
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        self.val_ops = tf.identity(total_loss, name='train_op')

      if type(self.clones[0].outputs) == list:
        self.val_ops.extend(self.clones[0].outputs)
      else:
        self.val_ops.append(self.clones[0].outputs)

      # Training saver
      self.saver = tf.train.Saver(var_list=variables_to_train, max_to_keep=2)

      # Global initialization
      self.sess.run(tf.global_variables_initializer())

      # Restore from checkpoint
      _get_init_fn(self, self.dump_dir)

  def infer_deploy(self, model):
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
                                                      clone_on_cpu=getattr(self, 'clone_on_cpu', False),
                                                      replica_id=0,
                                                      num_replicas=1,
                                                      num_ps_tasks=0)

      func = model.build
      @functools.wraps(func)
      def network_fn(*args, **kwargs):
        arg_scope = model.arg_scope()
        if arg_scope is not None:
          with slim.arg_scope(arg_scope):
            return func(is_training=self.is_training, *args, **kwargs)
        else:
          return func(is_training=self.is_training, *args, **kwargs)

      #######################
      # Create model clones #
      #######################
      self.clones = tfmodel_deploy.create_clones(deploy_config, network_fn)

      # Variables to train.
      variables_to_train = _get_variables_to_train(self)
      # Training saver
      self.saver = tf.train.Saver(var_list=variables_to_train)

      # Restore from checkpoint
      lastest_model_file = self.where_latest_model()
      self.saver.restore(self.sess, lastest_model_file)

      self.val_ops = self.clones[0].outputs
      if type(self.val_ops) != list:
        self.val_ops = [self.val_ops]

  # 1.step deploy model on hardwares
  def deploy(self, model):
    if self.is_training:
      self.training_deploy(model)
    else:
      self.infer_deploy(model)