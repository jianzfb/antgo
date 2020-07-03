# -*- coding: UTF-8 -*-
# @Time    : 10-10-18
# @File    : chapter_8_stargan.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.trainer.tfgantrainer import *
from chapter_8_ops import *
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
class StarGANModel(ModelDesc):
  def __init__(self):
    super(StarGANModel, self).__init__()

  def generator(self, x_input, c, reuse=False, scope="generator"):
    channel = ctx.params.ch
    c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
    c = tf.tile(c, [1, x_input.shape[1], x_input.shape[2], 1])
    x = tf.concat([x_input, c], axis=-1)

    with tf.variable_scope(scope, reuse=reuse):
      x = conv2d(x, channel, kernel_size=7, stride=1, use_bias=False, name='conv')
      x = instance_norm(x, name='ins_norm')
      x = relu(x)

      # 降采样
      for i in range(2):
        x = conv2d(x, channel * 2, kernel_size=4, stride=2, use_bias=False, name='conv_' + str(i))
        x = instance_norm(x, name='down_ins_norm_' + str(i))
        x = relu(x)

        channel = channel * 2

      for i in range(6):
        x = resblock(x, channel, use_bias=False, name='resblock_' + str(i))

      # 上采样
      for i in range(2):
        x = deconv2d(x, channel // 2, kernel_size=4, stride=2, use_bias=False, name='deconv_' + str(i))
        x = instance_norm(x, name='up_ins_norm' + str(i))
        x = relu(x)

        channel = channel // 2

      x = conv2d(x, 3, kernel_size=7, stride=1, use_bias=False, name='G_logit')
      x = tanh(x)

      return x

  def discriminator(self, x_input, reuse=False, scope="discriminator"):
    with tf.variable_scope(scope, reuse=reuse):
      channel = ctx.params.ch
      x = conv2d(x_input, channel, kernel_size=4, stride=2, use_bias=True, name='conv_0')
      x = lrelu(x, 0.01)

      for i in range(1, 6):
        x = conv2d(x, channel * 2, kernel_size=4, stride=2, use_bias=True, name='conv_' + str(i))
        x = lrelu(x, 0.01)

        channel = channel * 2

      c_kernel = int(ctx.params.load_size / np.power(2, 6))

      logit = conv2d(x, 1, kernel_size=3, stride=1, use_bias=False, name='D_logit')
      c = conv2d(x, ctx.params.label_num, kernel_size=c_kernel, stride=1, use_bias=False, name='D_label')
      c = tf.reshape(c, shape=[-1, ctx.params.label_num])

      return logit, c

  def gradient_panalty(self, real, fake, gan_type, scope="discriminator"):
    if gan_type == 'dragan':
      shape = tf.shape(real)
      eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
      x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
      x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
      noise = 0.5 * x_std * eps  # delta in paper

      # Author suggested U[0,1] in original paper, but he admitted it is bug in github
      # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

      alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
      interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X
    else:
      shape = tf.shape(real)
      alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
      interpolated = alpha * real + (1. - alpha) * fake

    logit, _ = self.discriminator(interpolated, reuse=True, scope=scope)

    GP = 0
    grad = tf.gradients(logit, interpolated)[0]           # gradient of D(interpolated)
    grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)  # l2 norm

    # WGAN - LP
    if gan_type == 'wgan-lp':
      GP = ctx.params.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

    elif gan_type == 'wgan-gp' or gan_type == 'dragan':
      GP = ctx.params.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

    return GP

  def model_fn(self,is_training=True, *args, **kwargs):
    if is_training:
      # 训练时，外部指定学习率
      self.lr = tf.placeholder(tf.float32, None, 'lr')

      # 训练时，外部喂入数据
      x_real = tf.placeholder(tf.float32,
                              [None, 128, 128, 3],
                              name='x_real')
      # 训练时，外部指定所喂入数据对应的属性标签
      x_label = tf.placeholder(tf.float32,
                               [None, ctx.params.label_num],
                               name='x_label')
      # 目标标签
      y_label = 1.0 - x_label

      # 定义生成器和判别器
      x_fake = self.generator(x_real, y_label)  # real a
      x_recon = self.generator(x_fake, x_label, reuse=True)  # real b

      real_logit, real_cls = self.discriminator(x_real)
      fake_logit, fake_cls = self.discriminator(x_fake, reuse=True)

      # 设置生成器和判别器 损失函数
      GP = 0
      if "wgan" in ctx.params.gan_type or ctx.params.gan_type == 'dragan':
        GP = self.gradient_panalty(real=x_real, fake=x_fake, gan_type=ctx.params.gan_type)
      else:
        GP = 0

      g_adv_loss = generator_loss(ctx.params.gan_type, fake=fake_logit)
      g_cls_loss = classification_loss(logit=fake_cls,
                                       label=tf.reshape(tf.tile(tf.reshape(y_label,[ctx.params.batch_size,
                                                                                    1,
                                                                                    1,
                                                                                    ctx.params.label_num]),[1,2,2,1]),
                                                        [-1,ctx.params.label_num]))
      g_rec_loss = L1_loss(x_real, x_recon)

      d_adv_loss = discriminator_loss(ctx.params.gan_type, real=real_logit, fake=fake_logit) + GP
      d_cls_loss = classification_loss(logit=real_cls,
                                       label=tf.reshape(tf.tile(tf.reshape(x_label,[ctx.params.batch_size,
                                                                                    1,
                                                                                    1,
                                                                                    ctx.params.label_num]),[1,2,2,1]),
                                                        [-1,ctx.params.label_num]))

      d_loss = ctx.params.adv_weight * d_adv_loss + \
               ctx.params.cls_weight * d_cls_loss
      d_loss = tf.identity(d_loss, name='d_loss')

      g_loss = ctx.params.adv_weight * g_adv_loss + \
               ctx.params.cls_weight * g_cls_loss + \
               ctx.params.rec_weight * g_rec_loss
      g_loss = tf.identity(g_loss,name='g_loss')

      return {g_loss: [x_fake, x_recon]}
    else:
      # 测试时，外部喂入数据
      x_real = tf.placeholder(tf.float32,
                              [1, 128, 128, 3],
                              name='x_real')

      # 测试时，外部喂入目标域属性标签
      y_label = tf.placeholder(tf.int32, [1, ctx.params.label_num], name='y_label')

      #
      y_fake = self.generator(x_real, y_label)  # real a

      return y_fake

##################################################
######## 3.step define training process  #########
##################################################
def preprocess_train_func(*args, **kwargs):
  img_A, img_B = args[0]

  is_a = True
  if np.random.random() > 0.5:
    is_a = False

  img = img_A
  label = [1, 0]
  if not is_a:
    img = img_B
    label = [0, 1]

  img = scipy.misc.imresize(img, [ctx.params.load_size, ctx.params.load_size])
  if len(img.shape) == 2:
    img = np.concatenate((np.expand_dims(img, -1), np.expand_dims(img, -1), np.expand_dims(img, -1)), axis=2)

  img = img[:,:,0:3]
  if np.random.random() > 0.5:
    img = np.fliplr(img)

  img = img / 127.5 - 1.
  return img, label

def training_callback(data_source, dump_dir):
  # 1. 创建GAN训练器
  tf_trainer = TFGANTrainer(dump_dir)

  # 2. 为GAN训练器加载模型 （这里采用StarGAN）
  tf_trainer.deploy(StarGANModel(),
                    d_loss={'scope': 'discriminator', 'learning_rate': 'lr'},
                    g_loss={'scope': 'generator', 'learning_rate': 'lr'})

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_train_func, Node.inputs(data_source))
  batch_node = BatchData(Node.inputs(preprocess_node), ctx.params.batch_size)

  # 4. 迭代训练（max_epochs 指定最大数据源重复次数）
  count = 0
  last_g_loss = 0.0
  for epoch in range(ctx.params.max_epochs):
    # 4.1. 自定义学习率下降策略
    lr_val = ctx.params.lr if epoch < ctx.params.epoch_step else ctx.params.lr * (ctx.params.max_epochs - epoch) / (ctx.params.max_epochs - ctx.params.epoch_step)

    # 4.2. 训练一次Generator网络，训练一次Discriminator网络
    for x_real, x_label in batch_node.iterator_value():
      # x_real:   [N x H x W x 3]
      # x_label:  [N x C]

      # 更新判别器网络
      d_loss = tf_trainer.d_loss_run(x_real=x_real,x_label=x_label,lr=lr_val)

      # 更新生成器网络
      g_loss = None
      if count % ctx.params.n_critic == 0:
        g_loss, x_fake, x_recon = tf_trainer.g_loss_run(x_real=x_real, x_label=x_label,lr=lr_val)
        last_g_loss = g_loss

      if g_loss is None:
        g_loss = last_g_loss

      # 每隔50步打印日志
      if count % 50 == 0:
        logger.info('g_loss %f d_loss %f at iterator %d in epoch %d (lr=%f)'%(g_loss, d_loss, count, epoch, lr_val))

      count += 1

    # 4.3. 保存模型
    tf_trainer.snapshot(epoch, count)


###################################################
######## 4.step define infer process     ##########
###################################################
def preprocess_test_func(*args, **kwargs):
  img_A = args[0]
  img_A = scipy.misc.imresize(img_A, [ctx.params.load_size, ctx.params.load_size])
  img_A = img_A / 127.5 - 1.
  img_A = np.expand_dims(img_A, 0)
  return img_A

def infer_callback(data_source, dump_dir):
  # 1. 创建GAN训练器
  tf_trainer = TFGANTrainer(dump_dir, False)

  # 2. 为GAN训练器加载模型（StarGANModel）
  tf_trainer.deploy(StarGANModel())

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_test_func, Node.inputs(data_source))

  # 4. 遍历所有数据 生成样本
  count = 0
  for a in preprocess_node.iterator_value():
    # 随机目标域标签
    y_label = np.array([0, 1]).reshape([1,2])
    fake_b = tf_trainer.run(x_real=a, y_label=y_label)

    fake_b = (fake_b + 1.) / 2.
    fake_b = np.squeeze(fake_b)
    if not os.path.exists(os.path.join(dump_dir, 'B')):
      os.makedirs(os.path.join(dump_dir, 'B'))

    # 4.1. 生成图
    scipy.misc.imsave(os.path.join(dump_dir, 'B', 'a2b_%d.png'%count), fake_b)

    # 4.2. 对应原图
    aa = (a + 1.) / 2.
    aa = np.squeeze(aa)
    scipy.misc.imsave(os.path.join(dump_dir, 'B', 'a%d.png' % count), aa)

    count += 1

    ctx.recorder.record({'RESULT': (fake_b*255).astype(np.uint8),'RESULT_TYPE': 'IMAGE'})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback