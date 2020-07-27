# -*- coding: UTF-8 -*-
# @Time    : 2020-06-05 16:04
# @File    : simple.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
# from antgo.trainer.tftrainer import *
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
from antgo.ant.debug import *
import cv2

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


class Tan(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(Tan, self).__init__(train_or_test, dir, params)
    # self.file_list = []
    # for f in os.listdir(dir):
    #   if f[0] == ".":
    #     continue
    #
    #   self.file_list.append(f)

  @property
  def size(self):
    if self.train_or_test == 'train':
      return 4

    if self.train_or_test == 'val':
      return 20

    if self.train_or_test == 'test':
      return 20

    return 0

  def split(self, split_params, split_method):
    return self, Tan('val', self.dir, self.ext_params)

  def data_pool(self):
    # for _ in range(self.size):
    #   yield np.random.randint(0,255,(100,100)), int(np.random.randint(0,3))
    for _ in range(self.size):
      yield np.random.randint(0, 255, (300, 100)), {'path': 'AA'}

##################################################
###### 1.1.step build visualization chart  #######
##################################################
# # every channel bind value type (NUMERIC, HISTOGRAM, IMAGE)
# loss_channel = ctx.job.create_channel("loss", "NUMERIC")
# ## bind channel to chart,every chart could include some channels
# ctx.job.create_chart([loss_channel], "Loss Curve", "step", "value")
# 测试数据可视化接口-1
loss1_channel = ctx.dashboard.create_channel("loss1", "LINE")
loss2_channel = ctx.dashboard.create_channel("loss2", "LINE")
loss_chart = ctx.dashboard.create_chart([loss1_channel, loss2_channel], 'A and B loss', "x", "y")

data_channel = ctx.dashboard.create_channel("data", "SCATTER")
data_chart = ctx.dashboard.create_chart([data_channel], 'data', 'x','y')

info_log = ctx.dashboard.create_channel("log", 'TEXT')
info_chart = ctx.dashboard.create_chart([info_log], 'log')

his1_channel = ctx.dashboard.create_channel('head distribution', 'BAR')
his2_channel = ctx.dashboard.create_channel('tail distribution', 'BAR')
his_chart = ctx.dashboard.create_chart([his1_channel, his2_channel], 'distribution compare')

heatmap_channel = ctx.dashboard.create_channel("WWYY", channel_type="HEATMAP")
heatmap_chart = ctx.dashboard.create_chart([heatmap_channel], 'DDDD', chart_x_axis="1,2,3,4", chart_y_axis="11,22,33,44")

table_channel = ctx.dashboard.create_channel("table", channel_type="TABLE")
table_chart = ctx.dashboard.create_chart([table_channel], 'table')


# 测试数据可视化接口-2



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
#     return 0 + self.candidates_size()
#
#   def split(self, split_params, split_method):
#     assert (split_method == 'holdout')
#     return self, MyDataset('val', self.dir, self.ext_params)
#
#   def data_pool(self):
#     # 1.step 训练数据
#
#     # 2.step 候选数据
#     for a,b in self.candidates():
#       yield a,b
#
#   def candidates(self, candidate_type='IMAGE'):
#     for a, b in super(MyDataset, self).candidates(candidate_type):
#       yield np.zeros((32, 32)), b
#
#   def unlabeled(self, tag=''):
#     count = 0
#     for a, b in super(MyDataset, self).unlabeled(tag):
#       print('get count %d'%count)
#       yield np.ones((32, 32)), b
#       count += 1
#
#   def check_candidate(self, unlabeled_files, candidate_folder):
#     # 检测标注数据是否符合要求
#     return True

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
# your model code
# class MNISTModel(ModelDesc):
#   def __init__(self):
#     super(MNISTModel, self).__init__()
#
#   def model_fn(self, is_training=True, *args, **kwargs):
#     image = tf.placeholder(tf.float32, [None, 784], name='image')
#     # 转换tensor形状到原始形状
#     feature = tf.reshape(image, [-1, 28, 28, 1])
#
#     # 第一组卷积
#     with tf.variable_scope('conv_layer1'):
#       h_conv1 = slim.conv2d(
#         feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
#       h_pool1 = tf.nn.max_pool(
#         h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     # from automl graph
#     # recommand_graph = Decoder().decode(self.automl['graph'])
#     # recommand_graph.layer_factory = LayerFactory()
#     # output_tensors = recommand_graph.materialization(input_nodes=[h_pool1])
#     # h_pool1 = output_tensors[0]
#
#     # 第二组卷积
#     with tf.variable_scope('conv_layer2'):
#       h_conv2 = slim.conv2d(
#         h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
#       h_pool2 = tf.nn.max_pool(
#         h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#       h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#
#     # 记录激活分布
#     tf.summary.histogram('activations', h_pool2)
#
#     # # 消除实验模块
#     # with ctx.block('auxilary_fc') as afc:
#     #   if afc.activate:
#     #     h_pool2_flat = slim.fully_connected(h_pool2_flat, 1024, activation_fn=tf.nn.relu)
#
#     # 第一组全连接（紧跟dropout模块）
#     h_fc1 = slim.dropout(
#       slim.fully_connected(
#         h_pool2_flat, 1024, activation_fn=tf.nn.relu),
#       keep_prob=0.5,
#       is_training=is_training)
#
#     # 第二组全连接
#     logits = slim.fully_connected(h_fc1, 10, activation_fn=None)
#
#     if is_training:
#       # 训练阶段，计算损失函数
#       label = tf.placeholder(tf.float32, [None], name='label')
#       target = tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0)
#       loss = tf.losses.softmax_cross_entropy(target, logits)
#       return loss
#     else:
#       # 推断阶段，返回计算概率值
#       logits = tf.nn.softmax(logits)
#       return logits


##################################################
######## 3.step define training process  #########
##################################################
# def action_func(*args, **kwargs):
#   a = args[0][0].reshape([-1])
#   b = args[0][1]['category_id']
#   return a, b
#

def training_callback(data_source, dump_dir):
  # logger.info('training model at {} stage'.format(ctx.stage))
  # for i in range(100):
  #   loss_channel.update(i, float(np.random.random()))
  #
  # text_1_channel.update(0, 'run')
  # text_1_channel.update(10, 'execute')
  # text_1_channel.update(30, 'sdf')
  # text_1_channel.update(31, 'run')
  # text_1_channel.update(32, 'execute')
  # text_1_channel.update(33, 'sdf')
  # text_1_channel.update(34, 'sdf')
  # text_1_channel.update(35, 'sdf')
  # text_1_channel.update(36, 'mm')
  # text_1_channel.update(37, 'sdf')

  log_text_list=['run','execute','sdf','run','mm']

  for i in range(100):
    # part 1 (线数据)
    loss1_channel.update(i, np.random.random() * 10)
    if i % 2 == 0:
      loss2_channel.update(i, np.random.random() * 10)

    # part 2 (scatter 数据)
    data_channel.update(i,i)

    # part 3 (日志数据)
    info_log.update(i, log_text_list[int(np.random.randint(0,5))])

    # part 3 (直方图 数据)
    his1_channel.update(i, np.random.randint(0,10,(20,40)))
    his2_channel.update(i, np.random.randint(0,5, (20,40)))

    # part 4 (热图)
    heatmap_channel.update(i, np.random.randint(0,10, (200,100)))

    # part 5 (TABLE)
    table_channel.update(i, [[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

    if i % 10 == 0:
      ctx.dashboard.update()

    time.sleep(1)





###################################################
######## 4.step define infer process     ##########
###################################################


def infer_callback(data_source, dump_dir):
  count = 0
  for ab in data_source.iterator_value():
    ctx.recorder.record({
      'RESULT': np.random.randint(0,3)
    })
    print(count)
    count += 1


def data_callback(data_source):
  for data in data_source.iterator_value():
    image, annotation = data
    yield {
      'data': image,
      'data_TYPE': 'IMAGE',
      'data_TAG': [],
      'data_ID': annotation['path'],
      'gt': image,
      'gt_TYPE': 'IMAGE',
      'gt_TAG': [],
      'gt_ID': annotation['path'],
      'gt1': image,
      'gt1_TYPE': 'IMAGE',
      'gt1_TAG': [],
      'gt1_ID': annotation['path'],
      'gt2': image,
      'gt2_TYPE': 'IMAGE',
      'gt2_TAG': [],
      'gt2_ID': annotation['path'],
      'gt3': image,
      'gt3_TYPE': 'IMAGE',
      'gt3_TAG': [],
      'gt3_ID': annotation['path'],
      'gt4': image,
      'gt4_TYPE': 'IMAGE',
      'gt4_TAG': [],
      'gt4_ID': annotation['path'],
      'gt5': image,
      'gt5_TYPE': 'IMAGE',
      'gt5_TAG': [],
      'gt5_ID': annotation['path'],
      'gt6': image,
      'gt6_TYPE': 'IMAGE',
      'gt6_TAG': [],
      'gt6_ID': annotation['path'],
      'gt7': image,
      'gt7_TYPE': 'IMAGE',
      'gt7_TAG': [],
      'gt7_ID': annotation['path'],
    }



###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
ctx.data_generator = data_callback

###################################################
###########    6.step test run         ############
###########                            ############
###################################################
if __name__ == '__main__':
  # # 1.step debug training process
  # debug_training_process(lambda :(np.zeros((28, 28)), {'category_id': 0}), 'mnist.yaml')
  #
  # # 2.step debug infer process
  # debug_infer_process(lambda :np.zeros((28,28)))
  pass