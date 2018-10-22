# -*- coding: UTF-8 -*-
# @Time    : 2018/10/15 3:27 PM
# @File    : chapter_7_perceptual.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.trainer.tftrainer import *
from chapter_7_ops import *
import vgg
from collections import namedtuple

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()

##################################################
###### 1.1.step build visualization chart  #######
##################################################
# every channel bind value type (NUMERIC, HISTOGRAM, IMAGE)
loss_channel = ctx.job.create_channel("loss", "NUMERIC")
## bind channel to chart,every chart could include some channels
ctx.job.create_chart([loss_channel], "Loss Curve", "step", "value")


##################################################
######## 2.step custom model building code  ######
##################################################
# your model code

##################################################
######## 2.1.step custom dataset parse code ######
##################################################
# your dataset parse code
# class MyDataset(Dataset):
#   def __init__(self, train_or_test, dir=None, params=None):
#     super(MyDataset, self).__init__(train_or_test, dir, params)
#
#   @property
#   def size(self):
#     return ...
#
#   def split(self, split_params, split_method):
#     assert (split_method == 'holdout')
#     return self, MyDataset('val', self.dir, self.ext_params)
#
#   def data_pool(self):
#     pass
#   def model_fn(self, *args, **kwargs):
#     # for tfrecords data
#     pass

##################################################
######## 2.2.step custom metric code        ######
##################################################
# your metric code
# class MyMeasure(AntMeasure):
#   def __init__(self, task):
#     super(MyMeasure, self).__init__(task, 'MyMeasure')
#
#   def eva(self, data, label):
#     return {'statistic': {'name': self.name,
#                           'value': [{'name': self.name, 'value': ..., 'type': 'SCALAR'}]}}


##################################################
######## 2.3.step custom model code        ######
##################################################
class PerceptualModel(ModelDesc):
  def __init__(self, style_image=None):
    super(PerceptualModel, self).__init__()
    self.style_image = style_image

  def gram_matrix_for_style_image(self, network, image, layer, style_image, sess):
      image_feature = network[layer].eval(feed_dict={image: style_image}, session=sess)
      image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
      return np.matmul(image_feature.T, image_feature) / image_feature.size

  def gram_matrix_for_input_image(self, network, layer):
      image_feature = network[layer]
      _, height, width, channels = image_feature.get_shape()

      size = int(height) * int(width) * int(channels)
      image_feature = tf.reshape(image_feature, (ctx.params.batch_size, height * width, channels))
      return tf.matmul(tf.transpose(image_feature, perm=[0,2,1]), image_feature) / float(size)

  def content_loss(self, styled_vgg, input_vgg, content_layer, content_weight):
    _, h, w, c = input_vgg[content_layer].get_shape()
    size = ctx.params.batch_size * int(h) * int(w) * int(c)

    return content_weight * (2 * tf.nn.l2_loss(input_vgg[content_layer] - styled_vgg[content_layer]) / float(size))

  def style_loss(self, styled_vgg, style_image, style_layers, style_weight, sess):
    style_image_placeholder = tf.placeholder('float', shape=style_image.shape)

    with slim.arg_scope(vgg.vgg_arg_scope(reuse=True)):
      _, style_loss_net = vgg.vgg_19(style_image_placeholder, num_classes=0, is_training=False)

    style_loss = 0
    style_preprocessed = style_image - np.array([ctx.params.R_MEAN,ctx.params.G_MEAN,ctx.params.B_MEAN]).reshape([1,1,1,3])

    for layer in style_layers:
      style_image_gram = self.gram_matrix_for_style_image(style_loss_net,
                                                          style_image_placeholder,
                                                          layer,
                                                          style_preprocessed,
                                                          sess)

      input_image_gram = self.gram_matrix_for_input_image(styled_vgg, layer)

      style_loss += (2 * tf.nn.l2_loss(input_image_gram - np.expand_dims(style_image_gram, 0)) / style_image_gram.size)
      return style_weight * style_loss

  def tv_loss(self, image, tv_weight):
    # total variation denoising
    shape = tuple(image.get_shape().as_list())

    _, h, w, c = image[:, 1:, :, :].get_shape()
    tv_y_size = ctx.params.batch_size * int(h) * int(w) * int(c)
    _, h, w, c = image[:, :, 1:, :].get_shape()
    tv_x_size = ctx.params.batch_size * int(h) * int(w) * int(c)
    tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :int(shape[1]) - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :int(shape[2]) - 1, :]) /
             tv_x_size))

    return tv_loss

  def image_transfer(self, image):
    image = image / 255.0
    conv1 = conv2d(image, 32, kernel_size=9, stride=1, name='conv1')
    conv1 = instance_norm(conv1, name='instance_norm1')
    conv1 = relu(conv1)

    conv2 = conv2d(conv1, 64, kernel_size=3, stride=2, name='conv2')
    conv2 = instance_norm(conv2, name='instance_norm2')
    conv2 = relu(conv2)

    conv3 = conv2d(conv2, 128, kernel_size=3, stride=2, name='conv3')
    conv3 = instance_norm(conv3, name='instance_norm3')
    conv3 = relu(conv3)

    resid1 = resblock(conv3, 128, 3, name='resblock1')
    resid2 = resblock(resid1, 128, 3, name='resblock2')
    resid3 = resblock(resid2, 128, 3, name='resblock3')
    resid4 = resblock(resid3, 128, 3, name='resblock4')
    resid5 = resblock(resid4, 128, 3, name='resblock5')

    conv_t1 = deconv2d(resid5, 64, kernel_size=3, stride=2, name='deconv1')
    conv_t1 = instance_norm(conv_t1, name='de_instance_norm1')
    conv_t1 = relu(conv_t1)

    conv_t2 = deconv2d(conv_t1, 32, kernel_size=3, stride=2, name='deconv2')
    conv_t2 = instance_norm(conv_t2, name='de_instance_norm2')
    conv_t2 = relu(conv_t2)

    conv_t3 = conv2d(conv_t2, 3, kernel_size=9, stride=1, name='last_conv')
    conv_t3 = instance_norm(conv_t3, name='last_instance_norm')

    preds = tf.nn.tanh(conv_t3)
    output = image + preds
    return tf.nn.tanh(output) * 127.5 + 255./2

  def model_fn(self, is_training=True, *args, **kwargs):
    # 输入图像占位
    # 数据范围 0 ~ 255
    input_batch = tf.placeholder(tf.float32,
                                 shape=[None, ctx.params.load_size, ctx.params.load_size, 3],
                                 name="image")

    # 对输入图像预处理
    input_batch = input_batch - np.array([ctx.params.R_MEAN, ctx.params.G_MEAN, ctx.params.B_MEAN]).reshape(
      [1, 1, 1, 3])

    # 构建风格化后图像
    with tf.variable_scope('transfer'):
      # 数据范围 0 ~ 255
      stylized_image = self.image_transfer(input_batch)

    if not is_training:
      # 在推断阶段仅返回风格化后的图片
      return stylized_image

    # 对风格化后的图像预处理
    stylized_image = stylized_image - np.array([ctx.params.R_MEAN, ctx.params.G_MEAN, ctx.params.B_MEAN]).reshape(
      [1, 1, 1, 3])

    # 加载标准vgg-19模型参数，并将输入图像和风格化图像分别送入vgg-19
    # 处理风格化后的图像stylized_image
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, stylied_vgg_output = vgg.vgg_19(stylized_image,
                                             is_training=False,
                                             num_classes=0)

    # 加载标准vgg-19模型参数
    kwargs['trainer'].sess.run(tf.global_variables_initializer())
    load_fn = slim.assign_from_checkpoint_fn('/Users/jian/Downloads/vgg_19.ckpt',
                                             slim.get_model_variables(scope='vgg_19'),
                                             ignore_missing_vars=True)
    load_fn(kwargs['trainer'].sess)

    # 处理输入图像 input_batch
    with slim.arg_scope(vgg.vgg_arg_scope(reuse=True)):
        _, input_vgg_output = vgg.vgg_19(input_batch,
                                           is_training=False,
                                           num_classes=0)

    # 内容损失函数 (输入图像和风格化后的图像之间的损失函数)
    content_loss = self.content_loss(stylied_vgg_output,
                                     input_vgg_output,
                                     'vgg_19/conv3/conv3_3',
                                     ctx.params.content_weight) / ctx.params.batch_size

    # 风格损失函数 （输入风格图和风格化后图像之间的损失函数）
    style_loss = self.style_loss(stylied_vgg_output,
                                 self.style_image,
                                 ['vgg_19/conv1/conv1_2',
                                  'vgg_19/conv2/conv2_2',
                                  'vgg_19/conv3/conv3_3',
                                  'vgg_19/conv4/conv4_3'],
                                 ctx.params.style_weight,
                                 kwargs['trainer'].sess) / ctx.params.batch_size

    # TV损失函数
    total_variation_loss = self.tv_loss(stylized_image, float(ctx.params.tv_weight)) / ctx.params.batch_size

    # 总损失函数
    loss = content_loss + style_loss + total_variation_loss
    slim.losses.add_loss(loss)

    return loss, stylized_image


##################################################
######## 3.step define training process  #########
##################################################
def preprocess_train_func(*args, **kwargs):
  image_a, _ = args[0]
  image_a = scipy.misc.imresize(image_a, [ctx.params.load_size, ctx.params.load_size])

  if len(image_a.shape) == 2:
    image_a = np.concatenate((np.expand_dims(image_a, -1),
                              np.expand_dims(image_a, -1),
                              np.expand_dims(image_a, -1)), axis=2)

  image_a = image_a[:,:,0:3]
  return image_a, None


def training_callback(data_source, dump_dir):
  # 1. 创建训练器
  tf_trainer = TFTrainer(dump_dir, True)

  # 2. 为训练器加载模型
  style_image = scipy.misc.imread('./starry-night-van-gogh.jpg')
  style_image = scipy.misc.imresize(style_image, [ctx.params.load_size, ctx.params.load_size])
  style_image = style_image.reshape([1, ctx.params.load_size, ctx.params.load_size, 3])
  tf_trainer.deploy(PerceptualModel(style_image))

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_train_func, Node.inputs(data_source))
  batch_node = BatchData(Node.inputs(preprocess_node), ctx.params.batch_size)

  # 4. 迭代训练
  for epoch in range(ctx.params.max_epochs):
    for a, _ in batch_node.iterator_value():
      # 4.1 运行一次迭代
      loss_val, stylyed_image = tf_trainer.run(image=a)

    # 4.2 保存模型
    tf_trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
def preprocess_infer_func(*args, **kwargs):
  image_a = args[0]
  image_a = scipy.misc.imresize(image_a, [ctx.params.load_size, ctx.params.load_size])

  if len(image_a.shape) == 2:
    image_a = np.concatenate((np.expand_dims(image_a, -1),
                              np.expand_dims(image_a, -1),
                              np.expand_dims(image_a, -1)), axis=2)

  image_a = image_a[:,:,0:3]
  return image_a


def infer_callback(data_source, dump_dir):
  # 1. 创建训练器
  tf_trainer = TFTrainer(dump_dir, False)

  # 2. 为训练器加载模型
  tf_trainer.deploy(PerceptualModel())

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_infer_func, Node.inputs(data_source))

  # 4. 遍历所有数据，并生成风格化图像
  for a in preprocess_node.iterator_value():
    styled_a = tf_trainer.run(image=np.expand_dims(a, 0))
    styled_a = np.squeeze(styled_a)
    ctx.recorder.record({'RESULT': styled_a.astype(np.uint8), 'RESULT_TYPE': 'IMAGE'})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback