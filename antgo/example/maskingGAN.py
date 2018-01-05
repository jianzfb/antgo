# -*- coding: UTF-8 -*-
# @Time    : 18-1-2
# @File    : maskingGAN.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from antgo.utils._resize import *
from antgo.codebook.tf.layers import *
import tensorflow.contrib.slim as slim
from datetime import datetime
from antgo.dataflow.imgaug.regular import *
from antgo.codebook.tf.dataset import *

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2.step model building (tensorflow) ######
##################################################
def _instance_norm(x):
  with tf.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
    offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
    out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
    
    return out


def _lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
  with tf.variable_scope(name):
    if alt_relu_impl:
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
      return f1 * x + f2 * abs(x)
    else:
      return tf.maximum(x, leak * x)


def _conv2d(input,
            o_channels,
            ks, stride,
            stddev=0.02,
            padding='VALID',
            name="conv2d",
            do_norm=True,
            do_relu=True,
            relufactor=0):
  with tf.variable_scope(name):
    conv = tf.contrib.layers.conv2d(
      input, o_channels, ks, stride, padding,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
      biases_initializer=tf.constant_initializer(0.0)
    )
    if do_norm:
      conv = _instance_norm(conv)
    
    if do_relu:
      if (relufactor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = _lrelu(conv, relufactor, "lrelu")
    
    return conv


def _deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
              stddev=0.02, padding="VALID", name="deconv2d",
              do_norm=True, do_relu=True, relufactor=0):
  with tf.variable_scope(name):
    
    conv = tf.contrib.layers.conv2d_transpose(
      inputconv, o_d, [f_h, f_w],
      [s_h, s_w], padding,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
      biases_initializer=tf.constant_initializer(0.0)
    )
    
    if do_norm:
      conv = _instance_norm(conv)
      # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
      # updates_collections=None, epsilon=1e-5, scale=True,
      # scope="batch_norm")
    
    if do_relu:
      if (relufactor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = _lrelu(conv, relufactor, "lrelu")
    
    return conv


def _resnet_block(inputres, dim, name="resnet", padding="REFLECT", use_dropout=False):
  """build a single block of resnet.

  :param inputres: inputres
  :param dim: dim
  :param name: name
  :param padding: for tensorflow version use REFLECT; for pytorch version use
   CONSTANT
  :return: a single block of resnet.
  """
  with tf.variable_scope(name):
    out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
    out_res = _conv2d(out_res, dim, 3, 1, 0.02, "VALID", "c1")
    if use_dropout:
      out_res = tf.nn.dropout(out_res, 0.5)
    out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
    out_res = _conv2d(out_res, dim, 3, 1, 0.02, "VALID", "c2", do_relu=False)
    
    return tf.nn.relu(out_res + inputres)


def generator(input, output_nc=4, name="generator", skip=False, use_dropout=True):
  with tf.variable_scope(name):
    f = 7
    ks = 3
    padding = "REFLECT"
    
    pad_input = tf.pad(input, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
    o_c1 = _conv2d(pad_input, ctx.params.ngf, f, 1, 0.02, name="c1")
    o_c2 = _conv2d(o_c1, ctx.params.ngf * 2, ks, 2, 0.02, "SAME", "c2")
    o_c3 = _conv2d(o_c2, ctx.params.ngf * 4, ks, 2, 0.02, "SAME", "c3")
    
    o_r1 = _resnet_block(o_c3, ctx.params.ngf * 4, "r1", padding, use_dropout)
    o_r2 = _resnet_block(o_r1, ctx.params.ngf * 4, "r2", padding, use_dropout)
    o_r3 = _resnet_block(o_r2, ctx.params.ngf * 4, "r3", padding, use_dropout)
    o_r4 = _resnet_block(o_r3, ctx.params.ngf * 4, "r4", padding, use_dropout)
    o_r5 = _resnet_block(o_r4, ctx.params.ngf * 4, "r5", padding, use_dropout)
    o_r6 = _resnet_block(o_r5, ctx.params.ngf * 4, "r6", padding, use_dropout)
    o_r7 = _resnet_block(o_r6, ctx.params.ngf * 4, "r7", padding, use_dropout)
    o_r8 = _resnet_block(o_r7, ctx.params.ngf * 4, "r8", padding, use_dropout)
    o_r9 = _resnet_block(o_r8, ctx.params.ngf * 4, "r9", padding, use_dropout)
    
    o_c4 = _deconv2d(
      o_r9, [ctx.params.batch_size, 128, 128, ctx.params.ngf * 2], ctx.params.ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
    o_c5 = _deconv2d(
      o_c4, [ctx.params.batch_size, 256, 256, ctx.params.ngf], ctx.params.ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
    o_c6 = _conv2d(o_c5, output_nc, f, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)
    
    if output_nc == 4:
      cc = tf.split(o_c6, 4, 3)
      mask = tf.nn.sigmoid(cc[0])
      image = tf.concat(cc[1:], 3)
      image = image * mask + input * (1-mask)
      o_c6 = image
      
    if skip is True:
      out_gen = tf.nn.tanh(input + o_c6, "t1")
    else:
      out_gen = tf.nn.tanh(o_c6, "t1")
    
    return out_gen


def discriminator(inputdisc, name="discriminator"):
  with tf.variable_scope(name):
    f = 4
    
    o_c1 = _conv2d(inputdisc, ctx.params.ndf, f, 2, 0.02, "SAME", "c1", relufactor=0.2)
    o_c2 = _conv2d(o_c1, ctx.params.ndf * 2, f, 2, 0.02, "SAME", "c2", relufactor=0.2)
    o_c3 = _conv2d(o_c2, ctx.params.ndf * 4, f, 2, 0.02, "SAME", "c3", relufactor=0.2)
    o_c4 = _conv2d(o_c3, ctx.params.ndf * 8, f, 2, 0.02, "SAME", "c4", relufactor=0.2)
    o_c5 = _conv2d(o_c4, 1, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)
    
    return o_c5


def cycle_consistency_loss(real_images, generated_images):
  """Compute the cycle consistency loss.

  The cycle consistency loss is defined as the sum of the L1 distances
  between the real images from each domain and their generated (fake)
  counterparts.

  This definition is derived from Equation 2 in:
      Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
      Networks.
      Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.


  Args:
      real_images: A batch of images from domain X, a `Tensor` of shape
          [batch_size, height, width, channels].
      generated_images: A batch of generated images made to look like they
          came from domain X, a `Tensor` of shape
          [batch_size, height, width, channels].

  Returns:
      The cycle consistency loss.
  """
  return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
  """Computes the LS-GAN loss as minimized by the generator.

  Rather than compute the negative loglikelihood, a least-squares loss is
  used to optimize the discriminators as per Equation 2 in:
      Least Squares Generative Adversarial Networks
      Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
      Stephen Paul Smolley.
      https://arxiv.org/pdf/1611.04076.pdf

  Args:
      prob_fake_is_real: The discriminator's estimate that generated images
          made to look like real images are real.

  Returns:
      The total LS-GAN loss.
  """
  return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
  """Computes the LS-GAN loss as minimized by the discriminator.

  Rather than compute the negative loglikelihood, a least-squares loss is
  used to optimize the discriminators as per Equation 2 in:
      Least Squares Generative Adversarial Networks
      Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
      Stephen Paul Smolley.
      https://arxiv.org/pdf/1611.04076.pdf

  Args:
      prob_real_is_real: The discriminator's estimate that images actually
          drawn from the real domain are in fact real.
      prob_fake_is_real: The discriminator's estimate that generated images
          made to look like real images are real.

  Returns:
      The total LS-GAN loss.
  """
  return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
          tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


class CycleGAN(object):
  """The CycleGAN module."""
  def __init__(self, pool_size, lambda_a, lambda_b, base_lr, skip=False):
    self._pool_size = pool_size

    self._lambda_a = lambda_a
    self._lambda_b = lambda_b
    self._base_lr = base_lr
    self._skip = skip
    
    self.fake_images_A = np.zeros(
      (self._pool_size, 1, ctx.params.img_height, ctx.params.img_width,
       ctx.params.img_channels)
    )
    self.fake_images_B = np.zeros(
      (self._pool_size, 1, ctx.params.img_height, ctx.params.img_width,
       ctx.params.img_channels)
    )
  
  def model_input(self):
    self.a = tf.placeholder(tf.uint8, shape=[None, None, 3])
    self.b = tf.placeholder(tf.uint8, shape=[None, None, 3])

    aa = tf.image.resize_images(
      self.a, [ctx.params.img_height, ctx.params.img_width])
    bb = tf.image.resize_images(
      self.b, [ctx.params.img_height, ctx.params.img_width])

    if ctx.params.do_flipping is True:
      aa = tf.image.random_flip_left_right(aa)
      bb = tf.image.random_flip_left_right(bb)
      
    aa = tf.subtract(tf.div(aa, 127.5), 1)
    bb = tf.subtract(tf.div(bb, 127.5), 1)
  
    self.aa = tf.expand_dims(aa, 0)
    self.bb = tf.expand_dims(bb, 0)
  
  def model_setup(self):
    """
    This function sets up the model to train.

    self.input_A/self.input_B -> Set of training images.
    self.fake_A/self.fake_B -> Generated images by corresponding generator
    of input_A and input_B
    self.lr -> Learning rate variable
    self.cyc_A/ self.cyc_B -> Images generated after feeding
    self.fake_A/self.fake_B to corresponding generator.
    This is use to calculate cyclic loss
    """
    self.input_a = tf.placeholder(tf.float32, [
                                    1,
                                    ctx.params.img_height,
                                    ctx.params.img_width,
                                    ctx.params.img_channels
                                  ], name="input_A")
    self.input_b = tf.placeholder(tf.float32,[
                                    1,
                                    ctx.params.img_height,
                                    ctx.params.img_width,
                                    ctx.params.img_channels
                                  ], name="input_B")
    
    self.fake_pool_A = tf.placeholder(tf.float32, [
                                        None,
                                        ctx.params.img_height,
                                        ctx.params.img_width,
                                        ctx.params.img_channels
                                      ], name="fake_pool_A")
    self.fake_pool_B = tf.placeholder(tf.float32, [
                                        None,
                                        ctx.params.img_height,
                                        ctx.params.img_width,
                                        ctx.params.img_channels
                                      ], name="fake_pool_B")
    
    self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

    images_a = self.input_a
    images_b = self.input_b
    fake_pool_a = self.fake_pool_A
    fake_pool_b = self.fake_pool_B
    
    # 1.step main model
    with tf.variable_scope("Model") as scope:
      self.prob_real_a_is_real = discriminator(images_a, "d_A")
      self.prob_real_b_is_real = discriminator(images_b, "d_B")
  
      self.fake_images_b = generator(images_a, output_nc=4, name="g_A", skip=self._skip)
      self.fake_images_a = generator(images_b, output_nc=4, name="g_B", skip=self._skip)
  
      scope.reuse_variables()
  
      self.prob_fake_a_is_real = discriminator(self.fake_images_a, "d_A")
      self.prob_fake_b_is_real = discriminator(self.fake_images_b, "d_B")
  
      self.cycle_images_a = generator(self.fake_images_b, output_nc=4, name="g_B", skip=self._skip)
      self.cycle_images_b = generator(self.fake_images_a, output_nc=4, name="g_A", skip=self._skip)
  
      scope.reuse_variables()
  
      self.prob_fake_pool_a_is_real = discriminator(fake_pool_a, "d_A")
      self.prob_fake_pool_b_is_real = discriminator(fake_pool_b, "d_B")
      
  def model_loss(self):
    cycle_consistency_loss_a = \
      self._lambda_a * cycle_consistency_loss(
        real_images=self.input_a, generated_images=self.cycle_images_a,
      )
    cycle_consistency_loss_b = \
      self._lambda_b * cycle_consistency_loss(
        real_images=self.input_b, generated_images=self.cycle_images_b,
      )
  
    lsgan_loss_a = lsgan_loss_generator(self.prob_fake_a_is_real)
    lsgan_loss_b = lsgan_loss_generator(self.prob_fake_b_is_real)
  
    self.g_loss_A = \
      cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
    self.g_loss_B = \
      cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a
  
    self.d_loss_A = lsgan_loss_discriminator(
      prob_real_is_real=self.prob_real_a_is_real,
      prob_fake_is_real=self.prob_fake_pool_a_is_real,
    )
    self.d_loss_B = lsgan_loss_discriminator(
      prob_real_is_real=self.prob_real_b_is_real,
      prob_fake_is_real=self.prob_fake_pool_b_is_real,
    )
  
    optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
  
    self.model_vars = tf.trainable_variables()
  
    d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
  
    self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
    self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
    self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
    self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)


##################################################
######## 3.step define training process  #########
##################################################
def _pool_fake_image(num_fakes, fake, fake_pool):
  if num_fakes < ctx.params.pool_size:
    fake_pool[num_fakes] = fake
    return fake
  else:
    p = random.random()
    if p > 0.5:
      random_id = random.randint(0, ctx.params.pool_size - 1)
      temp = fake_pool[random_id]
      fake_pool[random_id] = fake
      return temp
    else:
      return fake


def training_callback(data_source, dump_dir):
  ##########  1.step build model     ##############
  cycle_gan = CycleGAN(ctx.params.pool_size, ctx.params.lambda_a, ctx.params.lambda_b, ctx.params.base_lr)
  # 1.1.step building input model
  cycle_gan.model_input()
  # 1.2.step building model main root
  cycle_gan.model_setup()
  # 1.3.step building model loss
  cycle_gan.model_loss()
  
  ##########  2.step training model  ##############
  fake_images_A = np.zeros((ctx.params.pool_size,
                            1,
                            ctx.params.img_height,
                            ctx.params.img_width,
                            ctx.params.img_channels))

  fake_images_B = np.zeros((ctx.params.pool_size,
                            1,
                            ctx.params.img_height,
                            ctx.params.img_width,
                            ctx.params.img_channels))
  
  # record fake input number (from generator)
  num_fake_inputs = 0
  
  with tf.Session() as sess:
    # 2.1.step initializing model
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    sess.run(init)
    
    # save object
    saver = tf.train.Saver()

    iter = 0
    for epoch in range(ctx.params.max_epochs):
      # save model
      saver.save(sess, os.path.join(dump_dir, "maskingGAN"), global_step=epoch)
      # Dealing with the learning rate as per the epoch number
      if epoch < 100:
        curr_lr = ctx.params.base_lr
      else:
        curr_lr = ctx.params.base_lr - \
                  ctx.params.base_lr * (epoch - 100) / 400
      
      data_source._reset_iteration_state()
      for no_smiling, smiling in data_source.iterator_value():
        iter += 1
        # 2.2.step training process
        # 2.2.0.step data prepare
        aa, bb = sess.run([cycle_gan.aa, cycle_gan.bb], feed_dict={cycle_gan.a: smiling, cycle_gan.b: no_smiling})
        
        # 2.2.1.step optimizing the G_A network
        _, fake_B_temp, g_loss_A_val = sess.run([cycle_gan.g_A_trainer,
                                                 cycle_gan.fake_images_b,
                                                 cycle_gan.g_loss_A],
                                                feed_dict={
                                                  cycle_gan.input_a: aa,
                                                  cycle_gan.input_b: bb,
                                                  cycle_gan.learning_rate: curr_lr
                                                }
                                              )
        
        fake_B_temp1 = _pool_fake_image(num_fake_inputs, fake_B_temp, fake_images_B)
        
        # 2.2.2.step optimizing the D_B network
        _, d_loss_B_val = sess.run([cycle_gan.d_B_trainer, cycle_gan.d_loss_B],
                                    feed_dict={
                                      cycle_gan.input_a: aa,
                                      cycle_gan.input_b: bb,
                                      cycle_gan.learning_rate: curr_lr,
                                      cycle_gan.fake_pool_B: fake_B_temp1
                                    }
                                  )
        
        # 2.2.3.step optimizing the G_B network
        _, fake_A_temp, g_loss_B_val = sess.run([cycle_gan.g_B_trainer,
                                                 cycle_gan.fake_images_a,
                                                 cycle_gan.g_loss_B],
                                                feed_dict={
                                                  cycle_gan.input_a: aa,
                                                  cycle_gan.input_b: bb,
                                                  cycle_gan.learning_rate: curr_lr
                                                }
                                              )
        
        fake_A_temp1 = _pool_fake_image(num_fake_inputs, fake_A_temp, fake_images_A)
        
        # 2.2.4.step optimizing the D_A network
        _, d_loss_A_val = sess.run([cycle_gan.d_A_trainer, cycle_gan.d_loss_A],
                                    feed_dict={
                                      cycle_gan.input_a: aa,
                                      cycle_gan.input_b: bb,
                                      cycle_gan.learning_rate: curr_lr,
                                      cycle_gan.fake_pool_A: fake_A_temp1
                                    }
                                  )
        
        if iter % 100 == 0:
          logger.info('g_A_loss %f d_A_loss %f g_B_loss %f d_B_loss %f in iterator %d (epoch %d)' %
                      (g_loss_A_val, d_loss_A_val, g_loss_B_val, d_loss_B_val, iter, epoch))
        
        num_fake_inputs += 1


###################################################
######## 4.step define infer process     ##########
###################################################
import cv2
def infer_callback(data_source, dump_dir):
  #################################################
  data_queue = DatasetQueue(data_source, [tf.uint8], [(None, None, None)], max_queue_size=1)
  a = data_queue.dequeue()
  
  # Preprocessing:
  aa = tf.image.resize_images(a, [ctx.params.img_height, ctx.params.img_width])
  aa = tf.subtract(tf.div(aa, 127.5), 1)
  aa = tf.expand_dims(aa, 0)
  aa.set_shape((1, 256, 256, 3))
  
  ##########  1.step build model     ##############
  cycle_gan = CycleGAN(ctx.params.pool_size, ctx.params.lambda_a, ctx.params.lambda_b, ctx.params.base_lr)
  cycle_gan.model_setup()
  init = (tf.global_variables_initializer(),
          tf.local_variables_initializer())
  saver = tf.train.Saver()
  
  with tf.Session() as sess:
    sess.run(init)
    # chkpt_fname = tf.train.latest_checkpoint('/home/mi/20180103.223432.634196/train/cyclegan-100')
    chkpt_fname = '/home/mi/20180103.223432.634196/train/cyclegan-100'
    saver.restore(sess, chkpt_fname)
    
    coord = tf.train.Coordinator()
    custom_dataset_queue = tf.get_collection('CUSTOM_DATASET_QUEUE')
    custom_dataset_queue[0].coord = coord
    threads = custom_dataset_queue[0].start_threads(sess, 1)
    queue_threads = tf.train.start_queue_runners(coord=coord)
    threads.extend(queue_threads)
    
    for i in range(data_source.size):
      # 1.step input a
      aa_val = sess.run(aa)
      # 2.step transfer to b
      bb_val = sess.run(cycle_gan.fake_images_b, feed_dict={cycle_gan.input_a: aa_val})
      
      img_aa = np.squeeze(aa_val, 0)
      img_bb = np.squeeze(bb_val, 0)
      
      img_aa = ((img_aa + 1) * 127.5).astype(np.uint8)
      img_bb = ((img_bb + 1) * 127.5).astype(np.uint8)
      # cv2.imshow('aa', img_aa)
      # cv2.imshow('bb', img_bb)
      # cv2.waitKey()
      
      ctx.recorder.record(img_bb)
      cv2.imwrite('/home/mi/aa/%d-aa.png' % i, img_aa)
      cv2.imwrite('/home/mi/aa/%d-bb.png' % i, img_bb)
      print(i)
    
    coord.request_stop()
    coord.join(threads)


###################################################
####### 5.step custom dataset factor   ############
#######                                ############
###################################################
class CustomDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(CustomDataset, self).__init__(train_or_test, dir)
    assert (train_or_test in ['train', 'test'])
    
    # 1.step extract data list
    self.data_a_list = []
    self.data_b_list = []

    for a in os.listdir(os.path.join(self.dir, 'pos')):
      if a[0] != '.':
        self.data_a_list.append(os.path.join(self.dir, 'pos', a))
    
    for b in os.listdir(os.path.join(self.dir, 'neg')):
      if b[0] != '.':
        self.data_b_list.append(os.path.join(self.dir, 'neg', b))

    self.ids = list(range(len(self.data_a_list)))
  
  @property
  
  def size(self):
    return len(self.data_a_list)
  
  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1
      
      idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(idxs)
      
      for k in idxs:
        a = self.data_a_list[k]
        random_b_index = np.random.randint(0, len(self.data_b_list), 1)[0]
        b = self.data_b_list[random_b_index]
        a_img = imread(a)
        b_img = imread(b)
        yield [a_img, b_img]

###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
ctx.dataset_factory = lambda x: CustomDataset