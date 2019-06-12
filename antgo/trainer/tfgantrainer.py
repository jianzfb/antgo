# -*- coding: UTF-8 -*-
# Time: 10/07/18
# File: tfgantrainer.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import tensorflow.contrib.slim as slim
from antgo.trainer.trainer import *
from antgo.utils import logger
from antgo.measures.moving_statistic import *
from antgo.context import *
from antgo.trainer import tfmodel_deploy
import functools
import os
import re


class TFGANTrainer(Trainer):
  def __init__(self, dump_dir, is_training=True):
    super(TFGANTrainer, self).__init__(is_training)
    self.dump_dir = dump_dir
    self.time_stat = MovingAverage(self.log_every_n_steps)

    self.coord = None
    self.threads = None

    self.saver = None
    self.sess = None
    self.graph = None

    self.loss_list = {}
    self.update_list = {}
    self.lr_list = {}
    self.optimizer_list = {}
    self.var_list = {}
    self.global_step_list = {}

    self.trainop_list = {}
    self._has_model_input = False

    self.cache = {}
    self.loss_log = {}

    self.summary_op = None
    self.train_writer = None

    self.log_iter_at = {}

  def configure_optimizer(self, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if optimizer is not recognized.
    """
    if self.optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=self.adadelta_rho,
        epsilon=self.opt_epsilon)
    elif self.optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=self.adagrad_initial_accumulator_value)
    elif self.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=self.adam_beta1,
        beta2=self.adam_beta2,
        epsilon=self.opt_epsilon)
    elif self.optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=self.ftrl_learning_rate_power,
        initial_accumulator_value=self.ftrl_initial_accumulator_value,
        l1_regularization_strength=self.ftrl_l1,
        l2_regularization_strength=self.ftrl_l2)
    elif self.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=self.momentum,
        name='Momentum')
    elif self.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=self.rmsprop_decay,
        momentum=self.rmsprop_momentum,
        epsilon=self.opt_epsilon)
    elif self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer [%s] was not recognized', self.optimizer)
    return optimizer

  def get_variables_to_train(self):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    trainable_scopes = getattr(self, 'trainable_scopes', None)
    if trainable_scopes is None:
      trainable_filter = getattr(self, 'trainable_filter', None)
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

  def configure_learning_rate(self, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    batch_size = getattr(self, 'batch_size', 1) * getattr(self, 'num_clones', 1)
    num_epochs_per_decay = getattr(self, 'num_epochs_per_decay', 10)
    decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)

    sync_replicas = getattr(self, 'sync_replicas', False)
    if sync_replicas:
      replicas_to_aggregate = getattr(self, 'replicas_to_aggregate', 1)
      decay_steps /= replicas_to_aggregate

    learning_rate_decay_type = getattr(self, 'learning_rate_decay_type', 'exponential')
    learning_rate = getattr(self, 'learning_rate', 0.1)
    if learning_rate_decay_type == 'exponential':
      learning_rate_decay_factor = getattr(self, 'learning_rate_decay_factor', 0.9)
      return tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_steps,
                                        learning_rate_decay_factor,
                                        staircase=True,
                                        name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
      return tf.constant(learning_rate, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
      end_learning_rate = getattr(self, 'end_learning_rate', 0.0001)
      return tf.train.polynomial_decay(learning_rate,
                                       global_step,
                                       decay_steps,
                                       end_learning_rate,
                                       power=0.9,
                                       cycle=False,
                                       name='polynomial_decay_learning_rate')
    else:
      raise ValueError('learning_rate_decay_type [%s] was not recognized', learning_rate_decay_type)

  def get_init_fn(self, model, dump_dir, ctx=None):
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
          # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
          model_variables = tf.global_variables()
          exclusions = []
          checkpoint_exclude_scopes = getattr(self, 'checkpoint_exclude_scopes', None)
          if checkpoint_exclude_scopes is not None:
            exclusions = [scope.strip()
                          for scope in checkpoint_exclude_scopes.split(',')]

            logger.info('exclude scope %s from the latest checkpoint' % (','.join(exclusions)))

          fuzzy_exclusions = []
          checkpoint_fuzzy_exclude_scopes = getattr(self, 'checkpoint_fuzzy_exclude_scopes', None)
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
      logger.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % dump_dir)
      # initilize model from dump_dir
      latest_checkpoint = tf.train.latest_checkpoint(dump_dir)
      variables_to_restore = {}
      # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
      model_variables = tf.global_variables()

      for var in model_variables:
        var_name = var.op.name
        variables_to_restore[var_name] = var

      logger.info('load model from %s' % latest_checkpoint)
      return [slim.assign_from_checkpoint_fn(latest_checkpoint, variables_to_restore)]

    # 3.step load from checkpoint_path
    checkpoint_path = getattr(self, 'checkpoint_path', None)
    if checkpoint_path is None:
      return None

    exclusions = []
    checkpoint_exclude_scopes = getattr(self, 'checkpoint_exclude_scopes', None)
    if checkpoint_exclude_scopes is not None:
      exclusions = [scope.strip()
                    for scope in checkpoint_exclude_scopes.split(',')]

    fuzzy_exclusions = []
    checkpoint_fuzzy_exclude_scopes = getattr(self, 'checkpoint_fuzzy_exclude_scopes', None)
    if checkpoint_fuzzy_exclude_scopes is not None:
      fuzzy_exclusions = [scope.strip()
                          for scope in checkpoint_fuzzy_exclude_scopes.split(',')]

    transfers = {}
    checkpoint_transfer_scopes = getattr(self, 'checkpoint_transfer_scopes', None)
    if checkpoint_transfer_scopes is not None:
      for scope in checkpoint_transfer_scopes.split(','):
        s_scope, t_scope = scope.split(':')
        transfers[s_scope.strip()] = t_scope.strip()

    # TODO(sguada) variables.filter_variables()
    auxilary_variables_to_restore = []
    variables_to_restore = {}
    # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
    model_variables = tf.global_variables()
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
    return [slim.assign_from_checkpoint_fn(checkpoint_path, vr, ignore_missing_vars=False) for vr in
            auxilary_variables_to_restore if len(vr) > 0]

  def __getattr__(self, item):
    if not item.endswith('_run'):
      return getattr(super(TFGANTrainer, self), item)

    def func(**kwargs):
      kwargs.update({'loss_name': item.replace('_run','')})
      return self.run(**kwargs)

    return func

  def restore_scopy_from(self, model, restore_scope, checkpoint_path):
    # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
    model_variables = tf.global_variables()
    variables_to_restore = {}
    for var in model_variables:
      if var.op.name.startswith(restore_scope):
        variables_to_restore[var.op.name] = var

    if len(variables_to_restore) == 0:
      return

    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    logger.info('restore %s scope from %s' % checkpoint_path)
    fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore, ignore_missing_vars=False)
    fn(self.sess)


  def run(self, *args, **kwargs):
    assert (len(args) <= 1)

    data_generator = None
    if len(args) == 1:
      data_generator = args[0]

    loss_name = None
    if 'loss_name' in kwargs:
      loss_name = kwargs['loss_name']
      kwargs.pop('loss_name')

    if data_generator is not None:
      return self._run_by_generator(data_generator, loss_name, **kwargs)
    else:
      return self._run_by_feed(loss_name, feed_dict=None, **kwargs)

  def _run_by_generator(self, data_generator, loss_name, **kwargs):
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

      return self._run_by_feed(loss_name, feed_dict=feed_dict, **kwargs)

  def _run_by_feed(self, loss_name=None, feed_dict=None, **kwargs):
    if feed_dict is None:
      feed_dict = {}
      for k_name, v_value in kwargs.items():
        if k_name not in self.cache:
          k_tensor = self.graph.get_tensor_by_name('{}:0'.format(k_name))
          self.cache[k_name] = k_tensor

        feed_dict[self.cache[k_name]] = v_value

    start_time = time.time()
    loss_name = loss_name if loss_name is not None else 'GAN'
    result = self.sess.run(self.trainop_list[loss_name],
                           feed_dict=feed_dict if len(feed_dict) > 0 else None)
    elapsed_time = int((time.time() - start_time) * 100) / 100.0

    if self.ctx.recorder is not None and self.ctx.recorder.model_fn is not None:
      self.ctx.recorder.action(result[-1])
      result = result[:-1]

    # record iterator count
    self.iter_at += 1
    if loss_name not in self.log_iter_at:
      self.log_iter_at[loss_name] = 0
    self.log_iter_at[loss_name] += 1

    if self.summary_op is not None:
      summary_op_val = result[0]
      result = result[1:]
      self.train_writer.add_summary(summary_op_val, self.iter_at)

    # record elapsed time
    self.time_stat.add(elapsed_time)

    # print log
    if self.is_training:
      loss_val = 0.0
      if type(result) == list and len(result) >= 2:
        loss_val = result[1]
      elif type(result) == list and len(result) == 1:
        loss_val = result[0]
      else:
        loss_val = result

      # record loss value
      self.loss_log[loss_name].add(loss_val)

      #
      if self.log_iter_at[loss_name] % self.log_every_n_steps == 0:
        if not self.is_distribute_training or (self.is_distribute_training and self.rank == 0):
          loss_log_str = ''
          for k,v in self.loss_log.items():
            loss_log_str = loss_log_str + ' %s %f'%(k, v.get())

          logger.info('(PID: %s) INFO: %s lr %f at iterator %d (%f sec/step)' %
                      (str(os.getpid()),
                       loss_log_str,
                       self.sess.run(self.lr_list[loss_name]),
                       self.log_iter_at[loss_name],
                       float(self.time_stat.get())))
    else:
      if not self.is_distribute_training or (self.is_distribute_training and self.rank == 0):
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

  def training_deploy(self, model, *args,**kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
      # default graph
      self.graph = graph

      # session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      devices = self.ctx.devices if len(self.ctx.devices) > 0 else self.devices
      config.gpu_options.visible_device_list = ','.join(str(x) for x in devices) if len(devices) > 0 else ''
      self.sess = tf.Session(graph=graph, config=config)

      #######################
      # Config model deploy #
      #######################
      deploy_config = tfmodel_deploy.DeploymentConfig(num_clones=1,
                                                      devices=self.devices,
                                                      clone_on_cpu=self.clone_on_cpu,
                                                      replica_id=self.replica_id,
                                                      num_replicas=self.worker_replicas,
                                                      num_ps_tasks=self.num_ps_tasks)

      # init some info
      with tf.device(deploy_config.inputs_device()):
        #############################
        #### Define model input #####
        #############################
        with tf.variable_scope('input'):
          data_queue = self.ctx.model.model_input(self.is_training)
          if data_queue is not None:
            self._has_model_input = True

        func = model.model_fn
        @functools.wraps(func)
        def network_fn(*args, **kwargs):
          #
          logger.info('building computing graph')
          res = func(self.is_training, *args, **kwargs)
          tf.train.write_graph(self.sess.graph_def, self.dump_dir, 'graph.pbtxt')
          return res

        ####################################
        ####### Create summary      ########
        ####################################
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        #############################
        # Create model clones #######
        #############################
        self.clones = tfmodel_deploy.create_clones(deploy_config,
                                                   network_fn,
                                                   [data_queue] if data_queue is not None else None)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
          num_samples = self.num_samples if self.num_samples > 0 else self.ctx.data_source.size
          trainable_variables = self.get_variables_to_train()       # all
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # all

          for loss in tf.get_collection(tf.GraphKeys.LOSSES, deploy_config.clone_scope(0)):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

          for loss_name, loss_config in kwargs.items():
            loss_scope = loss_config['scope']

            # config loss log
            self.loss_log[loss_name] = MovingAverage(10)

            # Extract loss variable
            self.loss_list[loss_name] = graph.get_tensor_by_name('{}:0'.format(loss_name))

            if 'learning_rate' in loss_config:
              self.lr_list[loss_name] = graph.get_tensor_by_name('{}:0'.format(loss_config['learning_rate']))
              self.global_step_list[loss_name] = None
            else:
              global_step = tf.Variable(0, trainable=False)
              self.lr_list[loss_name] = self.configure_learning_rate(num_samples, global_step)
              self.global_step_list[loss_name] = global_step

            summaries.add(tf.summary.scalar('%s_learning_rate'%loss_name, self.lr_list[loss_name]))

            # config optimization procedure
            self.optimizer_list[loss_name] = self.configure_optimizer(self.lr_list[loss_name])

            # config related variables
            for loss_scope_name in loss_scope.split(','):
              if loss_name not in self.var_list:
                self.var_list[loss_name] = []
              self.var_list[loss_name].extend([var for var in trainable_variables if loss_scope_name in var.name])

            # get update ops
            for loss_scope_name in loss_scope.split(','):
              if loss_name not in self.update_list:
                self.update_list[loss_name] = []
              self.update_list[loss_name].extend([var for var in update_ops if loss_scope_name in var.name])


        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, deploy_config.clone_scope(0)))

        # Merge all summaries together.
        self.summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # Variables to train.
        for loss_name, loss_var in self.loss_list.items():
          optimizer = self.optimizer_list[loss_name]

          with tf.name_scope(self.clones[0].scope):
            with tf.device(self.clones[0].device):
              clone_grad = optimizer.compute_gradients(loss_var, var_list=self.var_list[loss_name])
              grad_update = optimizer.apply_gradients(clone_grad,global_step=self.global_step_list[loss_name] if self.global_step_list[loss_name] is not None else None)

              self.update_list[loss_name].append(grad_update)

              with tf.control_dependencies([tf.group(*self.update_list[loss_name])]):
                train_op = tf.identity(loss_var, name='train_op_%s'%loss_name)
                self.trainop_list[loss_name] = train_op

                if self.clones[0].outputs is not None:
                  if type(self.clones[0].outputs) == dict:
                    for k,v in self.clones[0].outputs.items():
                      if type(k) != str:
                        k = k.name.replace(':0','')

                      if k == loss_name:
                        self.trainop_list[loss_name] = [train_op]
                        if type(v) == list or type(v) == tuple:
                          self.trainop_list[loss_name].extend(list(v))
                        else:
                          self.trainop_list[loss_name].append(v)

                if type(self.trainop_list[loss_name]) != list:
                  self.trainop_list[loss_name] = [self.trainop_list[loss_name]]

                if self.summary_op is not None:
                  val_ops_temp = [self.summary_op]
                  val_ops_temp.extend(self.trainop_list[loss_name])
                  self.trainop_list[loss_name] = val_ops_temp

        # summary write
        if not os.path.exists(os.path.join(self.dump_dir, 'summary')):
          os.makedirs(os.path.join(self.dump_dir, 'summary'))

        self.train_writer = tf.summary.FileWriter(os.path.join(self.dump_dir, 'summary'), graph)

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
        # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
        self.saver = tf.train.Saver(max_to_keep=2)

        # Restore from checkpoint
        restore_fns = self.get_init_fn(model, self.dump_dir, self.ctx)
        if restore_fns is not None:
          for restore_fn in restore_fns:
            restore_fn(self.sess)

        # resotre from auxilary checkpoint
        for auxilary_scope, auxilary_checkpoint in self.auxilary_checkpoints.items():
          self.restore_scopy_from(model, auxilary_scope, auxilary_checkpoint)

  def infer_deploy(self, model, *args, **kwargs):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
      # Default graph
      self.graph = graph
      # Session
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      devices = self.ctx.devices if len(self.ctx.devices) > 0 else self.devices
      config.gpu_options.visible_device_list = ','.join(str(x) for x in devices) if len(devices) > 0 else ''
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
        data_queue = None
        if self.ctx.ant is not None:
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
        self.clones = tfmodel_deploy.create_clones(deploy_config, network_fn, [data_queue] if data_queue is not None else None)

        # write graph
        tf.train.write_graph(graph.as_graph_def(), self.dump_dir, 'infer_graph.pbtxt')

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
        restore_fns = self.get_init_fn(model, self.dump_dir, self.ctx)
        if restore_fns is not None:
          for restore_fn in restore_fns:
            restore_fn(self.sess)

        # model saver
        # model_variables = slim.get_model_variables() if model.model_variables is None else model.model_variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # snapshot
        self.snapshot(0)

        # Value ops
        self.trainop_list['GAN'] = self.clones[0].outputs
        if type(self.trainop_list['GAN']) != list and type(self.trainop_list['GAN']) != tuple:
          self.trainop_list['GAN'] = self.trainop_list['GAN']
        if type(self.trainop_list["GAN"]) == tuple:
          self.trainop_list['GAN'] = list(self.trainop_list['GAN'])

        # Append recorder model fn
        if self.ctx.recorder is not None and self.ctx.recorder.model_fn is not None:
          self.trainop_list['GAN'].append(self.ctx.recorder.model_fn)

  def deploy(self, model, *args, **kwargs):
    # model context
    self.ctx.model = model
    model.trainer = self

    if self.is_training:
      self.training_deploy(model, *args, **kwargs)
    else:
      self.infer_deploy(model, *args, **kwargs)

      if self.ctx.ant is None:
        logger.info('successfully deploy model')
        exit(0)

  def wait_until_clear(self):
    if self.coord is not None:
      self.coord.request_stop()
      self.coord.join(self.threads)