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
from ops import *
from module import *
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
class CycleGANModel(ModelDesc):
  def __init__(self):
    super(CycleGANModel, self).__init__()
    self.discriminator = discriminator
    self.generator = generator_resnet
    self.criterionGAN = mae_criterion
    self.batch_size = ctx.params.batch_size
    self.image_size = ctx.params.fine_size
    self.input_c_dim = ctx.params.input_nc
    self.output_c_dim = ctx.params.output_nc
    self.L1_lambda = ctx.params.L1_lambda

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

  if np.random.random() > 0.5:
    img_A = np.fliplr(img_A)
    img_B = np.fliplr(img_B)

  img_A = img_A / 127.5 - 1.
  img_B = img_B / 127.5 - 1.

  img_AB = np.concatenate((img_A, img_B), axis=2)
  return img_AB


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

  # 4. 迭代训练（max_epochs 指定最大数据源重复次数）
  count = 0
  for epoch in range(ctx.params.max_epochs):
    # 4.1. 自定义学习率下降策略
    lr_val = ctx.params.lr if epoch < ctx.params.epoch_step else ctx.params.lr * (ctx.params.max_epochs - epoch) / (ctx.params.max_epochs - ctx.params.epoch_step)

    # 4.2. 训练一次Generator网络，训练一次Discriminator网络
    for real_ab, _ in batch_node.iterator_value():
      # 更新 Generator 网络，并记录生成结果
      g_loss_val, fake_A, fake_B = tf_trainer.run_dict('g_loss',
                                                       real_A_and_B_images=real_ab,
                                                       lr=lr_val)

      # 更新 Discriminator 网络
      d_loss_val = tf_trainer.run_dict('d_loss',
                                       real_A_and_B_images=real_ab,
                                       fake_A_sample=fake_A,
                                       fake_B_sample=fake_B,
                                       lr=lr_val)
      # 每隔50步打印日志
      if count % 50 == 0:
        logger.info('g_loss %f d_loss %f at iterator %d in epoch %d (lr=%f)'%(g_loss_val, d_loss_val, count, epoch, lr_val))

      count += 1

    # 5. 保存模型
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
    fake_b, = tf_trainer.run_dict(loss_name=None, test_A=a)

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

    ctx.recorder.record([{'RESULT': (fake_b*255).astype(np.uint8),'RESULT_TYPE': 'IMAGE'}])


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback