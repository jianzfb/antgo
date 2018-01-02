# -*- coding: UTF-8 -*-
# @Time    : 17-12-20
# @File    : pix2pix.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from antgo.dataflow.common import *
from antgo.context import *
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from antgo.trainer.trainer import *
from antgo.trainer.tftrainer import *
from antgo.codebook.tf.preprocess import *
import numpy as np
import glob

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2 step define chart channel ###########
##################################################
# channel 1
generator_loss_channel = ctx.job.create_channel('generator-loss', 'NUMERIC')
gan_loss_channel = ctx.job.create_channel('gan-loss', 'NUMERIC')
l1_loss_channel = ctx.job.create_channel('l1-loss', 'NUMERIC')
discriminator_loss_channel = ctx.job.create_channel('discriminator-loss', "NUMERIC")
# chart 1
ctx.job.create_chart([generator_loss_channel, discriminator_loss_channel], 'loss', 'step', 'value')


##################################################
######## 3.step model building (tensorflow) ######
##################################################
def lrelu(x, a):
  with tf.name_scope("lrelu"):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2
    
    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
  with tf.variable_scope("batchnorm"):
    # this block looks like it has 3 inputs on the graph unless we do this
    input = tf.identity(input)
  
    channels = input.get_shape()[3]
    offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
    scale = tf.get_variable("scale", [channels], dtype=tf.float32,
      initializer=tf.random_normal_initializer(1.0, 0.02))
    mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
    variance_epsilon = 1e-5
    normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def conv(batch_input, out_channels, stride):
  with tf.variable_scope("conv"):
    in_channels = batch_input.get_shape()[3]
    filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
      initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
    #     => [batch, out_height, out_width, out_channels]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
    return conv

def deconv(batch_input, out_channels):
  with tf.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
      initializer=tf.random_normal_initializer(0, 0.02))
    # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
    #     => [batch, out_height, out_width, out_channels]
    conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
      [1, 2, 2, 1], padding="SAME")
    return conv

EPS = 1e-12


def create_generator(generator_inputs, generator_outputs_channels):
  layers = []
  
  # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
  with tf.variable_scope("encoder_1"):
    output = conv(generator_inputs, ctx.params.ngf, stride=2)
    layers.append(output)
  
  layer_specs = [
    ctx.params.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
    ctx.params.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
    ctx.params.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
    ctx.params.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
    ctx.params.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    ctx.params.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
    ctx.params.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
  ]
  
  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      rectified = lrelu(layers[-1], 0.2)
      # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
      convolved = conv(rectified, out_channels, stride=2)
      output = batchnorm(convolved)
      layers.append(output)
  
  layer_specs = [
    (ctx.params.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    (ctx.params.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    (ctx.params.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    (ctx.params.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    (ctx.params.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    (ctx.params.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    (ctx.params.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
  ]
  
  num_encoder_layers = len(layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        input = layers[-1]
      else:
        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
      
      rectified = tf.nn.relu(input)
      # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      output = batchnorm(output)
      
      if dropout > 0.0:
        output = tf.nn.dropout(output, keep_prob=1 - dropout)
      
      layers.append(output)
  
  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
  with tf.variable_scope("decoder_1"):
    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    layers.append(output)
  
  return layers[-1]


# def create_model(inputs, targets):
#   def create_discriminator(discrim_inputs, discrim_targets):
#     n_layers = 3
#     layers = []
#
#     # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
#     input = tf.concat([discrim_inputs, discrim_targets], axis=3)
#
#     # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
#     with tf.variable_scope("layer_1"):
#       convolved = conv(input, ctx.params.ndf, stride=2)
#       rectified = lrelu(convolved, 0.2)
#       layers.append(rectified)
#
#     # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
#     # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
#     # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
#     for i in range(n_layers):
#       with tf.variable_scope("layer_%d" % (len(layers) + 1)):
#         out_channels = ctx.params.ndf * min(2 ** (i + 1), 8)
#         stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
#         convolved = conv(layers[-1], out_channels, stride=stride)
#         normalized = batchnorm(convolved)
#         rectified = lrelu(normalized, 0.2)
#         layers.append(rectified)
#
#     # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
#     with tf.variable_scope("layer_%d" % (len(layers) + 1)):
#       convolved = conv(rectified, out_channels=1, stride=1)
#       output = tf.sigmoid(convolved)
#       layers.append(output)
#
#     return layers[-1]
#
#   with tf.variable_scope("generator") as scope:
#     out_channels = int(targets.get_shape()[-1])
#     outputs = create_generator(inputs, out_channels)
#
#   # create two copies of discriminator, one for real pairs and one for fake pairs
#   # they share the same underlying variables
#   with tf.name_scope("real_discriminator"):
#     with tf.variable_scope("discriminator"):
#       # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#       predict_real = create_discriminator(inputs, targets)
#
#   with tf.name_scope("fake_discriminator"):
#     with tf.variable_scope("discriminator", reuse=True):
#       # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
#       predict_fake = create_discriminator(inputs, outputs)
#
#   with tf.name_scope("discriminator_loss"):
#     # minimizing -tf.log will try to get inputs to 1
#     # predict_real => 1
#     # predict_fake => 0
#     discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
#
#   with tf.name_scope("generator_loss"):
#     # predict_fake => 1
#     # abs(targets - outputs) => 0
#     gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
#     gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
#     gen_loss = gen_loss_GAN * ctx.params.gan_weight + gen_loss_L1 * ctx.params.l1_weight
#
#   with tf.name_scope("discriminator_train"):
#     discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
#     discrim_optim = tf.train.AdamOptimizer(0.0002, 0.5)
#     discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
#     discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
#
#   gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#
#   with tf.name_scope("generator_train"):
#     with tf.control_dependencies([discrim_train]):
#       gen_loss = tf.identity(gen_loss)
#       tf.losses.add_loss(gen_loss)
#
#       ema = tf.train.ExponentialMovingAverage(decay=0.99)
#       update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
#       tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_losses)
#       return ema.average(gen_loss_GAN), ema.average(gen_loss_L1), ema.average(discrim_loss)
#
#   # with tf.name_scope("generator_train"):
#   #   with tf.control_dependencies([discrim_train]):
#   #     gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#   #     gen_optim = tf.train.AdamOptimizer(0.0002, 0.5)
#   #     gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#   #     gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
#   #
#   # ema = tf.train.ExponentialMovingAverage(decay=0.99)
#   # update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
#   #
#   # global_step = tf.contrib.framework.get_or_create_global_step()
#   # incr_global_step = tf.assign(global_step, global_step + 1)
#   # return tf.group(update_losses, incr_global_step, gen_train), ema.average(gen_loss_GAN), ema.average(gen_loss_L1), ema.average(discrim_loss)
#

def preprocess(image):
  with tf.name_scope("preprocess"):
    # [0, 1] => [-1, 1]
    return image * 2 - 1

CROP_SIZE = 256


def load_examples():
  if ctx.params.input_dir is None or not os.path.exists(ctx.params.input_dir):
    raise Exception("input_dir does not exist")

  input_paths = glob.glob(os.path.join(ctx.params.input_dir, "*.jpg"))
  decode = tf.image.decode_jpeg
  if len(input_paths) == 0:
    input_paths = glob.glob(os.path.join(ctx.params.input_dir, "*.png"))
    decode = tf.image.decode_png

  if len(input_paths) == 0:
    raise Exception("input_dir contains no image files")

  def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

  # if the image names are numbers, sort by the value rather than asciibetically
  # having sorted inputs means that the outputs are sorted in test mode
  if all(get_name(path).isdigit() for path in input_paths):
    input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
  else:
    input_paths = sorted(input_paths)

  with tf.name_scope("load_images"):
    path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)
    raw_input = decode(contents)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
  
    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
      raw_input = tf.identity(raw_input)
  
    raw_input.set_shape([None, None, 3])
  
    # break apart image pair and move to range [-1, 1]
    width = tf.shape(raw_input)[1]  # [height, width, channels]
    a_images = preprocess(raw_input[:, :width // 2, :])
    b_images = preprocess(raw_input[:, width // 2:, :])

  inputs, targets = [a_images, b_images]
  # synchronize seed for image operations so that we do the same operations to both
  # input and output images
  seed = random.randint(0, 2 ** 31 - 1)

  def transform(image):
    r = image
    r = tf.image.random_flip_left_right(r, seed=seed)
  
    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [ctx.params.scale_size, ctx.params.scale_size], method=tf.image.ResizeMethod.AREA)
  
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, ctx.params.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
    if ctx.params.scale_size > CROP_SIZE:
      r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    elif ctx.params.scale_size < CROP_SIZE:
      raise Exception("scale size cannot be less than crop size")
    return r

  with tf.name_scope("input_images"):
    input_images = transform(inputs)

  with tf.name_scope("target_images"):
    target_images = transform(targets)

  inputs_batch, targets_batch = tf.train.batch([input_images, target_images], batch_size=ctx.params.batch_size)
  steps_per_epoch = int(np.math.ceil(len(input_paths) / ctx.params.batch_size))

  return inputs_batch, targets_batch, steps_per_epoch


class Pix2PixModel(ModelDesc):
  def __init__(self):
    super(Pix2PixModel, self).__init__()
  
  def model_input(self, is_training, data_source):
    return load_examples()
  
  def model_fn(self, is_training=True, *args, **kwargs):
    inputs, targets, _ = args[0]

    def create_discriminator(discrim_inputs, discrim_targets):
      n_layers = 3
      layers = []
  
      # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
      input = tf.concat([discrim_inputs, discrim_targets], axis=3)
  
      # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
      with tf.variable_scope("layer_1"):
        convolved = conv(input, ctx.params.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)
  
      # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
      # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
      # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
      for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
          out_channels = ctx.params.ndf * min(2 ** (i + 1), 8)
          stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
          convolved = conv(layers[-1], out_channels, stride=stride)
          normalized = batchnorm(convolved)
          rectified = lrelu(normalized, 0.2)
          layers.append(rectified)
  
      # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
      with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)
  
      return layers[-1]

    with tf.variable_scope("generator") as scope:
      out_channels = int(targets.get_shape()[-1])
      outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
      with tf.variable_scope("discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
      with tf.variable_scope("discriminator", reuse=True):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
      # minimizing -tf.log will try to get inputs to 1
      # predict_real => 1
      # predict_fake => 0
      discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
      # predict_fake => 1
      # abs(targets - outputs) => 0
      gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
      gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
      gen_loss = gen_loss_GAN * ctx.params.gan_weight + gen_loss_L1 * ctx.params.l1_weight
      # model loss
      tf.losses.add_loss(gen_loss)

    with tf.name_scope("discriminator_train"):
      discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
      discrim_optim = tf.train.AdamOptimizer(0.0002, 0.5)
      discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
      discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    kwargs['trainer'].model_variables = gen_tvars
    # kwargs['trainer'].model_dependence = [discrim_train]
    #
    with tf.name_scope("generator_train"):
      with tf.control_dependencies([discrim_train]):
        gen_loss = tf.identity(gen_loss)
        tf.losses.add_loss(gen_loss)
    
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_losses)
    return ema.average(gen_loss_GAN), ema.average(gen_loss_L1), ema.average(discrim_loss)

##################################################
######## 4.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # config trainer
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.trainable_filter = 'generator'
  tf_trainer.deploy(Pix2PixModel())

  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size))
    for i in range(rounds):
      _, gen_loss_GAN, gen_loss_L1, discrim_loss = tf_trainer.run()
      
      if i % 100 == 0:
        logger.info('gan-loss: %f, gan-l1-loss: %f, dis-loss: %f in epoch %d step %d' % (gen_loss_GAN,
                                                                                         gen_loss_L1,
                                                                                         discrim_loss,
                                                                                         epoch,
                                                                                         i))

    tf_trainer.snapshot(epoch)
      
  # ss= Pix2PixModel()
  # # model_input = ss.model_input(True, data_source)
  # data = load_examples()
  # model_ops = ss.model_fn(True, data)
  #
  # iter = 0
  # with tf.Session() as sess:
  #   # Global initialization
  #   sess.run(tf.global_variables_initializer())
  #   sess.run(tf.local_variables_initializer())
  #
  #   coord = tf.train.Coordinator()
  #   threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  #
  #   for epoch in range(ctx.params.max_epochs):
  #     rounds = int(float(data_source.size) / float(ctx.params.batch_size))
  #     for i in range(rounds):
  #       _, loss_gan_val, loss_gan_l1, loss_dis = sess.run(model_ops)
  #       # if (iter + 1) % 100 == 0:
  #       #   generator_loss_channel.send(iter, loss_val)
  #       #   gan_loss_channel.send(iter, loss_gan_val)
  #       #   l1_loss_channel.send(iter, loss_gan_l1)
  #       #   discriminator_loss_channel.send(loss_dis)
  #
  #       if i % 100 == 0:
  #         logger.info('gan-loss: %f, gan-l1-loss: %f, dis-loss: %f in epoch %d iter %d'%(loss_gan_val, loss_gan_l1, loss_dis, epoch, i))
  #     iter = iter + 1
  #
  #     # # save
  #     # tf_trainer.snapshot(epoch)
  #
  #   coord.request_stop()
  #   coord.join(threads)


###################################################
######## 5.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=False)
  tf_trainer.trainable_filter = 'generator'
  tf_trainer.deploy(Pix2PixModel())
  
  for _ in range(data_source.size):
    fake_output = tf_trainer.run()
    fake_output = np.squeeze(fake_output, 0)
    
    ctx.recorder.record(fake_output)


###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback