# -*- coding: UTF-8 -*-
# @Time    : 10-10-18
# @File    : chapter_8_munitgan.py
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
class MUNITModel(ModelDesc):
  def __init__(self):
    super(MUNITModel, self).__init__()

  ##################################################################################
  # Encoder and Decoders
  ##################################################################################
  def style_encoder(self, x, reuse=False, scope='style_encoder'):
      # IN removes the original feature mean and variance that represent important style information
      channel = ctx.params.ch
      with tf.variable_scope(scope, reuse=reuse) :
          x = conv2d(x, channel, kernel_size=7, stride=1, use_bias=True, name='conv_0')
          x = relu(x)

          for i in range(2) :
              x = conv2d(x, channel*2, kernel_size=4, stride=2, use_bias=True, name='conv_'+str(i+1))
              x = relu(x)

              channel = channel * 2

          for i in range(2) :
              x = conv2d(x, channel, kernel_size=4, stride=2, use_bias=True, name='down_conv_'+str(i))
              x = relu(x)

          x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True) # global average pooling
          x = conv2d(x, ctx.params.style_dim, kernel_size=1, stride=1, use_bias=True, name='SE_logit')

          return x

  def content_encoder(self, x, reuse=False, scope='content_encoder'):
      channel = ctx.params.ch
      with tf.variable_scope(scope, reuse=reuse) :
          x = conv2d(x, channel, kernel_size=7, stride=1, use_bias=True, name='conv_0')
          x = instance_norm(x, name='ins_0')
          x = relu(x)

          for i in range(ctx.params.n_downsample) :
              x = conv2d(x, channel*2, kernel_size=4, stride=2, use_bias=True, name='conv_'+str(i+1))
              x = instance_norm(x, name='ins_'+str(i+1))
              x = relu(x)

              channel = channel * 2

          for i in range(ctx.params.n_res) :
              x = resblock(x, channel, use_bias=True, name='resblock_'+str(i))

          return x

  def generator(self, contents, style, reuse=False, scope="decoder"):
      channel = ctx.params.mlp_dim
      with tf.variable_scope(scope, reuse=reuse) :
          mu, sigma = self.mlp_for_mu_sigma(style, reuse)
          x = contents
          for i in range(ctx.params.n_res) :
              x = adaptive_resblock(x, channel, mu, sigma, name='adaptive_resblock'+str(i))

          for i in range(ctx.params.n_upsample) :
              _, h, w, _ = x.get_shape().as_list()
              x = tf.image.resize_nearest_neighbor(x,size=[int(h) * 2, int(w) * 2])
              x = conv2d(x, channel//2, kernel_size=5, stride=1, use_bias=True, name='conv_'+str(i))
              x = instance_norm(x, name='instance_norm_'+str(i))
              x = relu(x)

              channel = channel // 2

          x = conv2d(x, ctx.params.img_ch, kernel_size=7, stride=1, use_bias=True, name='G_logit')
          x = tanh(x)

          return x

  def mlp_for_mu_sigma(self, style, reuse=False, scope='MLP'):
      with tf.variable_scope(scope, reuse=reuse) :
          x = linear(style, ctx.params.mlp_dim, name='linear_0')
          x = relu(x)

          x = linear(x, ctx.params.mlp_dim, name='linear_1')
          x = relu(x)

          mu = linear(x, ctx.params.mlp_dim, name='mu')
          sigma = linear(x, ctx.params.mlp_dim, name='sigma')

          mu = tf.reshape(mu, shape=[-1, 1, 1, ctx.params.mlp_dim])
          sigma = tf.reshape(sigma, shape=[-1, 1, 1, ctx.params.mlp_dim])

          return mu, sigma

  ##################################################################################
  # Discriminator
  ##################################################################################
  def discriminator(self, x_init, reuse=False, scope="discriminator"):
      D_logit = []
      with tf.variable_scope(scope, reuse=reuse) :
          for scale in range(ctx.params.n_scale) :
              channel = ctx.params.ch
              x = conv2d(x_init, channel, kernel_size=4, stride=2, use_bias=True, name='ms_' + str(scale) + 'conv_0')
              x = lrelu(x, 0.2)

              for i in range(1, ctx.params.n_dis):
                  x = conv2d(x, channel * 2, kernel_size=4, stride=2, use_bias=True, name='ms_' + str(scale) +'conv_' + str(i))
                  x = lrelu(x, 0.2)

                  channel = channel * 2

              x = conv2d(x, 1, kernel_size=1, stride=1, use_bias=True, name='ms_' + str(scale) + 'D_logit')
              D_logit.append(x)

              x_init = slim.avg_pool2d(x_init, kernel_size=3)

          return D_logit

  ##################################################################################
  # Model
  ##################################################################################
  def encoder_A(self, x_A, reuse=False):
      style_A = self.style_encoder(x_A, reuse=reuse, scope='style_encoder_A')
      content_A = self.content_encoder(x_A, reuse=reuse, scope='content_encoder_A')

      return content_A, style_A

  def encoder_B(self, x_B, reuse=False):
      style_B = self.style_encoder(x_B, reuse=reuse, scope='style_encoder_B')
      content_B = self.content_encoder(x_B, reuse=reuse, scope='content_encoder_B')

      return content_B, style_B

  def decoder_A(self, content_B, style_A, reuse=False):
      x_ba = self.generator(contents=content_B, style=style_A, reuse=reuse, scope='decoder_A')

      return x_ba

  def decoder_B(self, content_A, style_B, reuse=False):
      x_ab = self.generator(contents=content_A, style=style_B, reuse=reuse, scope='decoder_B')

      return x_ab

  def discriminate_real(self, x_A, x_B):
      real_A_logit = self.discriminator(x_A, scope="discriminator_A")
      real_B_logit = self.discriminator(x_B, scope="discriminator_B")

      return real_A_logit, real_B_logit

  def discriminate_fake(self, x_ba, x_ab):
      fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
      fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

      return fake_A_logit, fake_B_logit

  def model_fn(self, is_training=True, *args, **kwargs):
    if is_training:
      # 构建训练阶段网络
      self.lr = tf.placeholder(tf.float32, name='lr')
      # 来自域A的图像
      domain_A = tf.placeholder(tf.float32, shape=[None, ctx.params.img_h, ctx.params.img_w, 3], name='image_a')
      # 来自域B的图像
      domain_B = tf.placeholder(tf.float32, shape=[None, ctx.params.img_h, ctx.params.img_w, 3], name='image_b')

      # 域A风格
      style_a = tf.placeholder(tf.float32, shape=[None, 1, 1, ctx.params.style_dim], name='style_a')
      # 域B风格
      style_b = tf.placeholder(tf.float32, shape=[None, 1, 1, ctx.params.style_dim], name='style_b')

      # 对来自域A的图像进行编码，得到内容编码和风格编码（对于风格编码的使用借鉴图像风格化的想法）
      content_a, style_a_prime = self.encoder_A(domain_A)
      # 对来自域B的图形进行编码，得到内容编码和风格编码
      content_b, style_b_prime = self.encoder_B(domain_B)

      # 使用A的内容编码和风格编码对域A图像进行重建（自身重建）
      # Decoder相当于生成器，这里使用来自VAE的概念
      x_aa = self.decoder_A(content_B=content_a, style_A=style_a_prime)
      # 使用B的内容编码和风格编码对域B图像进行重建（自身重建）
      x_bb = self.decoder_B(content_A=content_b, style_B=style_b_prime)

      # 使用B的内容编码和A的风格编码，生成符合域A风格的B图（跨域重建）
      x_ba = self.decoder_A(content_B=content_b, style_A=style_a, reuse=True)
      # 使用A的内容编码和B的风格编码，生成符合域B风格的A图（跨域重建）
      x_ab = self.decoder_B(content_A=content_a, style_B=style_b, reuse=True)

      # 借鉴来自cyclegan的想法（循环一致性损失），对跨域重建图进行重新编码
      content_b_, style_a_ = self.encoder_A(x_ba, reuse=True)
      content_a_, style_b_ = self.encoder_B(x_ab, reuse=True)

      x_aba = self.decoder_A(content_B=content_a_, style_A=style_a_prime, reuse=True)
      x_bab = self.decoder_B(content_A=content_b_, style_B=style_b_prime, reuse=True)

      # 重新生成的A域图和真实A域图的一致性损失
      cyc_recon_A = L1_loss(x_aba, domain_A)
      cyc_recon_B = L1_loss(x_bab, domain_B)

      # 构建域A和域B的判别器网络
      real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)
      fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

      # 根据GAN类型，构建生成器的损失（多尺度下的损失）
      G_ad_loss_a = generator_loss_list(ctx.params.gan_type, fake_A_logit)
      G_ad_loss_b = generator_loss_list(ctx.params.gan_type, fake_B_logit)

      # 根据GAN类型，构建判别器的损失（多尺度下的损失）
      D_ad_loss_a = discriminator_loss_list(ctx.params.gan_type, real_A_logit, fake_A_logit)
      D_ad_loss_b = discriminator_loss_list(ctx.params.gan_type, real_B_logit, fake_B_logit)

      # 重建图像内容的损失
      recon_A = L1_loss(x_aa, domain_A)
      recon_B = L1_loss(x_bb, domain_B)

      # 编码风格损失的一致性损失
      recon_style_A = L1_loss(style_a_, style_a)
      recon_style_B = L1_loss(style_b_, style_b)

      # 编码内容损失的一致性损失
      recon_content_A = L1_loss(content_a_, content_a)
      recon_content_B = L1_loss(content_b_, content_b)

      # 生成器A的损失函数
      Generator_A_loss = ctx.params.gan_w * G_ad_loss_a + \
                         ctx.params.recon_x_w * recon_A + \
                         ctx.params.recon_s_w * recon_style_A + \
                         ctx.params.recon_c_w * recon_content_A + \
                         ctx.params.recon_x_cyc_w * cyc_recon_A

      # 生成器B的损失函数
      Generator_B_loss = ctx.params.gan_w * G_ad_loss_b + \
                         ctx.params.recon_x_w * recon_B + \
                         ctx.params.recon_s_w * recon_style_B + \
                         ctx.params.recon_c_w * recon_content_B + \
                         ctx.params.recon_x_cyc_w * cyc_recon_B

      # 判别器A的损失函数
      Discriminator_A_loss = ctx.params.gan_w * D_ad_loss_a
      # 判别器B的损失函数
      Discriminator_B_loss = ctx.params.gan_w * D_ad_loss_b

      # 生成器的整体损失（来自A和B）
      g_loss = Generator_A_loss + Generator_B_loss
      g_loss = tf.identity(g_loss, name='g_loss')

      # 判别器的整体损失（来自A和B）
      d_loss = Discriminator_A_loss + Discriminator_B_loss
      d_loss = tf.identity(d_loss, name='d_loss')

      return {'g_loss': [x_ab, x_ba]}
    else:
      domain_A = tf.placeholder(tf.float32, shape=[None, ctx.params.img_h, ctx.params.img_w, 3], name='image_a')
      style_b = tf.placeholder(tf.float32, shape=[None, 1, 1, ctx.params.style_dim], name='style_b')

      content_a, style_a_prime = self.encoder_A(domain_A)
      x_ab = self.decoder_B(content_A=content_a, style_B=style_b)

      return x_ab

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

  img_A = img_A[:,:,0:3]
  img_B = img_B[:,:,0:3]

  if np.random.random() > 0.5:
    img_A = np.fliplr(img_A)
    img_B = np.fliplr(img_B)

  img_A = img_A / 127.5 - 1.
  img_B = img_B / 127.5 - 1.

  return img_A, img_B


def training_callback(data_source, dump_dir):
  # 1. 创建GAN训练器
  tf_trainer = TFGANTrainer(dump_dir)

  # 2. 为GAN训练器加载模型 （MUNITModel）
  tf_trainer.deploy(MUNITModel(),
                    d_loss={'scope': 'discriminator', 'learning_rate': 'lr'},
                    g_loss={'scope': 'encoder,decoder', 'learning_rate': 'lr'})

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_train_func, Node.inputs(data_source))
  batch_node = BatchData(Node.inputs(preprocess_node), ctx.params.batch_size)

  # 4. 迭代训练（max_epochs 指定最大数据源重复次数）
  count = 0
  for epoch in range(ctx.params.max_epochs):
    # 4.1. 自定义学习率下降策略
    lr_val = ctx.params.lr if epoch < ctx.params.epoch_step else ctx.params.lr * (ctx.params.max_epochs - epoch) / (ctx.params.max_epochs - ctx.params.epoch_step)

    # 4.2. 训练一次Generator网络，训练一次Discriminator网络
    for image_a, image_b in batch_node.iterator_value():
      # x_real:   [N x H x W x 3]
      # x_label:  [N x C]
      style_a = np.random.normal(loc=0.0, scale=1.0, size=[ctx.params.batch_size, 1, 1, ctx.params.style_dim])
      style_b = np.random.normal(loc=0.0, scale=1.0, size=[ctx.params.batch_size, 1, 1, ctx.params.style_dim])

      # 更新判别器网络
      d_loss = tf_trainer.d_loss_run(style_a=style_a,
                                     style_b=style_b,
                                     image_a=image_a,
                                     image_b=image_b,
                                     lr=lr_val)

      # 更新生成器网络
      g_loss,fake_ab, fake_ba = tf_trainer.g_loss_run(style_a=style_a,
                                                      style_b=style_b,
                                                      image_a=image_a,
                                                      image_b=image_b,
                                                      lr=lr_val)

      # 每隔50步打印日志
      if count % ctx.params.log_step == 0:
        logger.info('g_loss %f d_loss %f at iterator %d in epoch %d (lr=%f)'%(g_loss, d_loss, count, epoch, lr_val))

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

  # 2. 为GAN训练器加载模型（StarGANModel）
  tf_trainer.deploy(MUNITModel())

  # 3. 预处理数据管道
  preprocess_node = Node('preprocess', preprocess_test_func, Node.inputs(data_source))

  # 4. 遍历所有数据 生成样本
  count = 0
  for a in preprocess_node.iterator_value():
    style_b = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, ctx.params.style_dim])
    fake_b = tf_trainer.run(image_a=a, style_b=style_b)

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