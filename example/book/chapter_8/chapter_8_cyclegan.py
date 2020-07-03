# -*- coding: UTF-8 -*-
# @Time    : 10-07-18
# @File    : chapter_8_cyclegan.py
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
def abs_criterion(in_, target):
  return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
  return tf.reduce_mean((in_ - target) ** 2)


class CycleGANModel(ModelDesc):
  def __init__(self):
    super(CycleGANModel, self).__init__()
    # self.discriminator = discriminator
    # self.generator = generator_resnet
    self.criterionGAN = mae_criterion
    self.batch_size = ctx.params.batch_size
    self.image_size = ctx.params.fine_size
    self.input_c_dim = ctx.params.input_nc
    self.output_c_dim = ctx.params.output_nc
    self.L1_lambda = ctx.params.L1_lambda

  def discriminator(self, image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4

  def generator(self, image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
      # image is 256 x 256 x input_c_dim
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      def residule_block(x, dim, ks=3, s=1, name='res'):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
        return y + x

      # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
      # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
      # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
      c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
      c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
      c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
      c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

      # define G network with 9 resnet blocks
      r1 = residule_block(c3, options.gf_dim * 4, name='g_r1')
      r2 = residule_block(r1, options.gf_dim * 4, name='g_r2')
      r3 = residule_block(r2, options.gf_dim * 4, name='g_r3')
      r4 = residule_block(r3, options.gf_dim * 4, name='g_r4')
      r5 = residule_block(r4, options.gf_dim * 4, name='g_r5')
      r6 = residule_block(r5, options.gf_dim * 4, name='g_r6')
      r7 = residule_block(r6, options.gf_dim * 4, name='g_r7')
      r8 = residule_block(r7, options.gf_dim * 4, name='g_r8')
      r9 = residule_block(r8, options.gf_dim * 4, name='g_r9')

      d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
      d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
      d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
      d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
      d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
      pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
      return pred

  def model_fn(self, is_training=True, *args, **kwargs):
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                           gf_dim df_dim output_c_dim is_training')
    self.options = OPTIONS._make((ctx.params.batch_size, ctx.params.fine_size,
                                  ctx.params.ngf, ctx.params.ndf, ctx.params.output_nc,
                                  is_training))
    if is_training:
      self.lr = tf.placeholder(tf.float32, None, 'lr')
      self.real_data = tf.placeholder(tf.float32,
                                      [None, self.image_size, self.image_size,
                                       self.input_c_dim + self.output_c_dim],
                                      name='real_A_and_B_images')

      self.real_A = self.real_data[:, :, :, :self.input_c_dim]
      self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

      self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
      self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
      self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
      self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

      self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
      self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
      self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                        + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                        + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
      self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                        + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                        + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
      self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                    + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                    + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                    + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
      self.g_loss = tf.identity(self.g_loss, 'g_loss')

      self.fake_A_sample = tf.placeholder(tf.float32,
                                          [None, self.image_size, self.image_size,
                                           self.input_c_dim], name='fake_A_sample')
      self.fake_B_sample = tf.placeholder(tf.float32,
                                          [None, self.image_size, self.image_size,
                                           self.output_c_dim], name='fake_B_sample')
      self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
      self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
      self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
      self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

      self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
      self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
      self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

      self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
      self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
      self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
      self.d_loss = self.da_loss + self.db_loss
      self.d_loss = tf.identity(self.d_loss, 'd_loss')

      return {'g_loss': [self.fake_A, self.fake_B]}
    else:
      self.test_A = tf.placeholder(tf.float32,
                                   [None, self.image_size, self.image_size,
                                    self.input_c_dim], name='test_A')
      # self.test_B = tf.placeholder(tf.float32,
      #                              [None, self.image_size, self.image_size,
      #                               self.output_c_dim], name='test_B')
      self.testB = self.generator(self.test_A, self.options, False, name="generatorA2B")
      # self.testA = self.generator(self.test_B, self.options, False, name="generatorB2A")
      return self.testB



##################################################
######## 3.step define training process  #########
##################################################
def preprocess_train_func(*args, **kwargs):
  img_A, img_B = args[0]

  img_A = scipy.misc.imresize(img_A, [ctx.params.load_size, ctx.params.load_size])
  img_B = scipy.misc.imresize(img_B, [ctx.params.load_size, ctx.params.load_size])
  h1 = int(np.ceil(np.random.uniform(1e-2, ctx.params.load_size - ctx.params.fine_size)))
  w1 = int(np.ceil(np.random.uniform(1e-2, ctx.params.load_size - ctx.params.fine_size)))
  img_A = img_A[h1:h1 + ctx.params.fine_size, w1:w1 + ctx.params.fine_size]
  img_B = img_B[h1:h1 + ctx.params.fine_size, w1:w1 + ctx.params.fine_size]

  if len(img_A.shape) == 2:
    img_A = np.concatenate((np.expand_dims(img_A, -1), np.expand_dims(img_A, -1), np.expand_dims(img_A, -1)), axis=2)

  if len(img_B.shape) == 2:
    img_B = np.concatenate((np.expand_dims(img_B, -1), np.expand_dims(img_B, -1), np.expand_dims(img_B, -1)), axis=2)

  img_A = img_A[:, :, 0:3]
  img_B = img_B[:, :, 0:3]

  if np.random.random() > 0.5:
    img_A = np.fliplr(img_A)
    img_B = np.fliplr(img_B)

  img_A = img_A / 127.5 - 1.
  img_B = img_B / 127.5 - 1.

  img_AB = np.concatenate((img_A, img_B), axis=2)
  return img_AB


class ImagePool(object):
  def __init__(self, maxsize=50):
    self.maxsize = maxsize
    self.num_img = 0
    self.images = []

  def __call__(self, image):
    if self.maxsize <= 0:
      return image
    if self.num_img < self.maxsize:
      self.images.append(image)
      self.num_img += 1
      return image
    if np.random.rand() > 0.5:
      idx = int(np.random.rand() * self.maxsize)
      tmp1 = copy.copy(self.images[idx])[0]
      self.images[idx][0] = image[0]
      idx = int(np.random.rand() * self.maxsize)
      tmp2 = copy.copy(self.images[idx])[1]
      self.images[idx][1] = image[1]
      return [tmp1, tmp2]
    else:
      return image


def training_callback(data_source, dump_dir):
  # 1. 创建GAN训练器
  tf_trainer = TFGANTrainer(dump_dir)

  # 2. 为GAN训练器加载模型（这里采用CycleGAN模型）
  tf_trainer.deploy(CycleGANModel(),
                    d_loss={'scope': 'discriminator', 'learning_rate': 'lr'},
                    g_loss={'scope': 'generator', 'learning_rate': 'lr'})

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_train_func, Node.inputs(data_source))
  batch_node = BatchData(Node.inputs(preprocess_node), ctx.params.batch_size)

  pool = ImagePool()

  # 4. 迭代训练（max_epochs 指定最大数据源重复次数）
  count = 0
  for epoch in range(ctx.params.max_epochs):
    # 4.1. 自定义学习率下降策略
    lr_val = ctx.params.lr if epoch < ctx.params.epoch_step else ctx.params.lr * (ctx.params.max_epochs - epoch) / (ctx.params.max_epochs - ctx.params.epoch_step)

    # 4.2. 训练一次Generator网络，训练一次Discriminator网络
    for real_ab, _ in batch_node.iterator_value():
      # 更新 Generator 网络，并记录生成结果
      g_loss_val, fake_A, fake_B = tf_trainer.g_loss_run(real_A_and_B_images=real_ab, lr=lr_val)

      [fake_A, fake_B] = pool([fake_A, fake_B])
      # 更新 Discriminator 网络
      d_loss_val = tf_trainer.d_loss_run(real_A_and_B_images=real_ab,
                                         fake_A_sample=fake_A,
                                         fake_B_sample=fake_B,
                                         lr=lr_val)
      # 每隔50步打印日志
      if count % 50 == 0:
        logger.info('g_loss %f d_loss %f at iterator %d in epoch %d (lr=%f)'%(g_loss_val, d_loss_val, count, epoch, lr_val))

      count += 1

    # 4.3. 保存模型
    tf_trainer.snapshot(epoch, count)


###################################################
######## 4.step define infer process     ##########
###################################################
def preprocess_test_func(*args, **kwargs):
  img_A = args[0]
  img_A = scipy.misc.imresize(img_A, [ctx.params.fine_size, ctx.params.fine_size])
  img_A = img_A / 127.5 - 1.
  img_A = np.expand_dims(img_A, 0)
  return img_A

def infer_callback(data_source, dump_dir):
  # 1. 创建GAN训练器
  tf_trainer = TFGANTrainer(dump_dir, False)

  # 2. 为GAN训练器加载模型（这里采用CycleGAN模型）
  tf_trainer.deploy(CycleGANModel())

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_test_func, Node.inputs(data_source))

  # 4. 遍历所有数据 生成样本
  count = 0
  for a in preprocess_node.iterator_value():
    fake_b = tf_trainer.run(test_A=a)

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
    ctx.recorder.record([{'RESULT': (fake_b*255).astype(np.uint8),'RESULT_TYPE': 'IMAGE'}])


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback