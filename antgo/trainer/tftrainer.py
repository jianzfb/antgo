# -*- coding: UTF-8 -*-
# File: tftrainer.py
# Author: jian(jian@mltalker.com)
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
import os
import re
from .trainer import *
from antgo.utils import logger
import numpy as np


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _optimization_op(optimization_config,lr):
    optimization_method = optimization_config['method']
    if optimization_method == 'RMS':
        decay = optimization_config.get('decay',0.9)
        momentum = optimization_config.get('momentum',0.0)
        epsilon = optimization_config.get('epsilon',1e-10)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr,decay=decay,momentum=momentum,epsilon=epsilon)
        return optimizer
    elif optimization_method == 'RSD':
        optimizer = tf.train.GradientDescentOptimizer(lr)
        return optimizer
    elif optimization_method == 'Adam':
        beta1 = optimization_config.get('beta1',0.9)
        beta2 = optimization_config.get('beta2',0.999)
        epsilon = optimization_config.get('epsilon',1e-8)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2,epsilon=epsilon)
        return optimizer
    elif optimization_method == 'Momentum':
        momentum = optimization_config.get('momentum', 0.0)
        optimizer = tf.train.MomentumOptimizer(lr,momentum)
        return optimizer
    else:
        return tf.train.GradientDescentOptimizer(lr)


def _lr_decay_op(*args, **kwargs):
    decay_rate = kwargs['decay_rate']
    decay_steps = kwargs['decay_steps']
    lr = kwargs['lr']
    staircase = kwargs['staircase']
    global_step = kwargs['global_step']
    return tf.train.exponential_decay(lr,global_step,decay_steps,decay_rate,staircase)


class TFTrainer(Trainer):
    def __init__(self, trainer_context,dump_dir,is_training=True):
        super(TFTrainer,self).__init__(trainer_context,is_training)
        self.dump_dir = dump_dir

    # 2.step run model once
    def run(self,data_generator,binds):
        # bind data
        feed_dict = {}
        feed_data = []
        current_graph = self.training_graph if self.is_training else self.infer_graph
        current_sess = self.training_sess if self.is_training else self.infer_sess
        with current_graph.as_default():
            for device_index in range(len(self.device_list)):
                # generate data
                data = next(data_generator)
                feed_data.append(data)

                # bind data
                tower_name = 'tower{}'.format(device_index) if device_index == 0 else 'towerp{}'.format(device_index - 1)
                with TowerContext(tower_name):
                    for k,v in binds.items():
                        placeholder_tensor = get_current_tower_context().find_tensor_in_tower(current_graph,k)
                        assert(placeholder_tensor is not None)
                        feed_dict[placeholder_tensor] = data[v]

            # modify running info
            self.iter_at += 1

            # running
            reorganized_feed_data = []
            for k in zip(*feed_data):
                reorganized_feed_data.append(k)

            feed_data = reorganized_feed_data
            # father method
            super(TFTrainer,self).run(data_generator,binds)
            return feed_data, current_sess.run(self.val_ops,feed_dict)

    # 3.step snapshot running state
    def snapshot(self, epoch=0):
        assert(self.is_training)
        logger.info('snapshot at %d in %d epoch' % (self.iter_at, epoch))
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        model_filename = "{prefix}_{infix}_iter_{d}.ckpt".format(prefix=self.snapshot_prefix,
                                                                 infix=self.snapshot_infix, d=self.iter_at)
        model_filepath = os.path.join(self.dump_dir,model_filename)
        self.training_saver.save(self.training_sess, model_filepath)

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

    def training_deploy(self,model):
        # basic config
        self.training_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.training_sess = tf.Session(graph=self.training_graph, config=config)

        tower_grads_and_vars = []
        tower_losses = []
        tower_saveble_vars = None
        tower_extra_train_ops = []

        # training graph
        with self.training_graph.as_default(), tf.device('/cpu:0'):
            # config trainer
            with tf.variable_scope(None, 'trainer'):
                # global_step op
                global_step = tf.get_variable(
                    'trainer_global_step', [], initializer=tf.constant_initializer(0), trainable=False)

                # decay op (for learning rate)
                lr_decay_op = _lr_decay_op(decay_rate=self.decay_rate,
                                        decay_steps=self.decay_steps,
                                        staircase=self.staircase,
                                        lr=self.lr,
                                        global_step=global_step)

                optimization_op = _optimization_op(self.optimization, lr_decay_op)

            # deploy model on hardwares
            logger.info('deploy model on hardwares')
            device_config = '/{prefix}:{index}'
            for device_index in range(len(self.device_list)):
                with tf.device(device_config.format(prefix=self.device_prefix,index=self.device_list[device_index])):
                    # tower name
                    tower_name = 'tower{}'.format(device_index) if device_index == 0 else 'towerp{}'.format(device_index - 1)
                    # tower context
                    with TowerContext(tower_name, is_training=True):
                        with tf.variable_scope('model') as model_scope:
                            if device_index > 0:
                                # share variables
                                model_scope.reuse_variables()

                            # op flow graph
                            loss_and_ext_train_op = model.build()
                            loss = None
                            ext_train_op = []
                            if type(loss_and_ext_train_op) == tuple or \
                                            type(loss_and_ext_train_op) == list:
                                loss, ext_train_op = loss_and_ext_train_op
                            else:
                                loss = loss_and_ext_train_op

                            # 1.step record saveble vars
                            if type(model.saveble_vars) == list:
                                if tower_saveble_vars is None:
                                    tower_saveble_vars = []
                                tower_saveble_vars.extend(model.saveble_vars)
                            else:
                                if tower_saveble_vars is None:
                                    tower_saveble_vars = {}
                                tower_saveble_vars.update(model.saveble_vars)

                            # 2.step record loss
                            tower_losses.append(loss)

                            # 3.step compute trainable vars gradient
                            # trainable var list under tower
                            # (trainable vars are the save under different tower)
                            grads_and_vars = optimization_op.compute_gradients(loss, tf.trainable_variables())
                            clip_grads_and_vars = []
                            for grad, var in grads_and_vars:
                                if grad is not None:
                                    clip_grads_and_vars.append((tf.clip_by_value(grad, self.min_grad, self.max_grad),
                                                                var))

                            # 3.1 step record tower grads
                            tower_grads_and_vars.append(clip_grads_and_vars)

                            # 4.step extra training ops
                            tower_extra_train_ops.extend(ext_train_op)

            # Training saver
            self.training_saver = tf.train.Saver(var_list=tower_saveble_vars, max_to_keep=2)

            # Average gradients,loss,and incorrect among towers
            average_grads_and_vars = average_gradients(tower_grads_and_vars)
            average_train_loss = tf.reduce_mean(tower_losses)

            # Apply the gradients to adjust the shared variables
            train_op = optimization_op.apply_gradients(average_grads_and_vars, global_step=global_step)
            train_ops = [train_op] + tower_extra_train_ops
            train_ops_group = tf.group(*train_ops)

            # Initialization All Variables
            self.training_sess.run(tf.global_variables_initializer())

            # Loading pre-trained Model
            if self.pre_trained_model is not None:
                for model_name, model_file in self.pre_trained_model.items():
                    needed_to_load_vars = model.graph_vars(model_name)
                    if needed_to_load_vars is not None:
                        logger.info('loading %s for subgraph %s'%(model_file,model_name))
                        pre_model_saver = tf.train.Saver(var_list=needed_to_load_vars)
                        pre_model_saver.restore(self.training_sess,model_file)

            # Loading Latest Model
            lastest_model_file = self.where_latest_model()
            if lastest_model_file:
                logger.info('loading latest model %s'%lastest_model_file)
                self.training_saver.restore(self.training_sess,lastest_model_file)

        # standard output operation after every iterator
        self.val_ops = [train_ops_group, average_train_loss, global_step, lr_decay_op]

    def infer_deploy(self,model):
        # basic config
        self.infer_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.infer_sess = tf.Session(graph=self.infer_graph, config=config)

        tower_saveble_vars = None
        tower_results = []

        # infer graph
        with self.infer_graph.as_default(), tf.device('/cpu:0'):
            # deploy model on hardwares
            logger.info('deploy model on hardwares')
            device_config = '/{prefix}:{index}'
            for device_index in range(len(self.device_list)):
                with tf.device(device_config.format(prefix=self.device_prefix,index=self.device_list[device_index])):
                    # tower name
                    tower_name = 'tower{}'.format(device_index) if device_index == 0 else 'towerp{}'.format(device_index - 1)
                    # tower context
                    with TowerContext(tower_name, is_training=False):
                        with tf.variable_scope('model') as model_scope:
                            if device_index > 0:
                                # share variables
                                model_scope.reuse_variables()

                            # op flow graph
                            result = model.build()

                            # 1.step record saveble vars (for load latest model)
                            if type(model.saveble_vars) == list:
                                if tower_saveble_vars is None:
                                    tower_saveble_vars = []
                                tower_saveble_vars.extend(model.saveble_vars)
                            else:
                                if tower_saveble_vars is None:
                                    tower_saveble_vars = {}
                                tower_saveble_vars.update(model.saveble_vars)

                            # 2.step record result
                            tower_results.append(result)

            # Initialization All Variables
            self.infer_sess.run(tf.global_variables_initializer())

            # load model from dump_dir
            lastest_model_file = self.where_latest_model()
            if lastest_model_file:
                logger.info('load latest model %s' % lastest_model_file)
                infer_saver = tf.train.Saver(var_list=tower_saveble_vars)
                infer_saver.restore(self.infer_sess,lastest_model_file)

        self.val_ops = tower_results

    # 1.step deploy model on hardwares
    def deploy(self, model):
        if self.is_training:
            self.training_deploy(model)
        else:
            self.infer_deploy(model)

    def watch(self, names, fuzzy=True):
        # find watch variables
        current_graph = self.training_graph if self.is_training else self.infer_graph
        watch_vars = TowerContext.find_tensor(current_graph,names,fuzzy)

        # add into val_ops'
        self.val_ops.extend(watch_vars)

    def clear(self):
        current_sess = self.training_sess if self.is_training else self.infer_sess
        current_sess.close()