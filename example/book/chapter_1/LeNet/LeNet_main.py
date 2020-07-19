# -*- coding: UTF-8 -*-
# @Time    : 2019-06-04
# @File    : LeNet_main.py
# @Author  : xxx
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.measures.base import *
from antgo.trainer.trainer import *
from antgo.ant.debug import *

from antgo.trainer.tftrainer import *
from antgo.codebook.tf.dataset import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
Antgo Machine Learning Running Framework Mechanism
+---------------------------------------------------------------+
|                                                               |
|                                                    MLTALKER   |
|                                                               |
| +----------------------------+          Experiment Manager    |
| | AntGO                      |                                |
| +--------+                   |          Experiment Analyse    |
| ||Control|      Task         |                                |
| +-+------+                   |          Experiment Vis        |
| | |   Datasource    Measure  |                                |
| | |       +   +        +     |                                |
| | |       |or |        |     |                   ^            |
+---------------------------------------------------------------+
  | |       |   v        |     |                   |
  | |       |  framework |     |                   |
  | |       |  dataflow  |     |                   |
  | |       |  pipeline  |     |                   |
  | |       |   +        |     |                   |
  | |       v   v        |     |                   |
  | | Framework Model    |     |                   |
  | |    optimize        |     |                   |
  | |       +            |     |                   |
  | v       v            v     |                   |
  | +-------+------------+-------------------------^
  |                            |      Experiment Info
  |                            |  (performance, and others)
  |Customised Module           |
  |   xx_main.py               |
  +----------------------------+

'''

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
###### 1.1.step build visualization chart  #######
##################################################
# every channel bind value type (NUMERIC, HISTOGRAM, IMAGE)
loss_channel = ctx.dashboard.create_channel("loss","LINE")
## bind channel to chart,every chart could include some channels
ctx.dashboard.create_chart([loss_channel], "Loss Curve", "step", "value")


##################################################
######## 2.step custom model building code  ######
##################################################


##################################################
######## 2.1.step custom dataset parse code ######
########    write dataset parse code if     ######
#####  you want to parse your private dataset ####
##################################################
'''
class LeNetDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(LeNetDataset, self).__init__(train_or_test, dir, params)
    # fill dataset
    self.dataset = []
    self.dataset_size = len(self.dataset)

    # fill dataset annotation
    self.label = []

  @property
  def size(self):
    # return data size
    return self.dataset_size

  def split(self, split_params, split_method):
    # set how to split dataset into val/test
    assert (split_method == 'holdout')
    return self, LeNetDataset('val', self.dir, self.ext_params)

  def data_pool(self):
    for i in range(self.size):
      yield self.dataset[i], self.label[i]
    
'''

##################################################
######## 2.2.step custom metric code        ######
##################################################
'''
class LeNetMeasure(AntMeasure):
  def __init__(self, task):
    super(LeNetMeasure, self).__init__(task, 'LeNetMeasure')
    self.is_support_rank = True

  def eva(self, data, label):
    if label is not None:
        data = zip(data, label)

    for predict, gt in data:
      # predict come from your model
      # gt come from annotation
      pass

    value = 0.0
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': value, 'type': 'SCALAR'}]}}
'''

##################################################
######## 2.3.step custom model code        ######
##################################################

'''
there are tow ways to feed data into tensorflow framework.
method 1:
+-----------------------------------------------+
|---------------+             +-----------------|
||local dataflow+------------>+tensorflow model||
||              |             |     (feed)     ||
|---------------+             +-----------------|
+-----------------------------------------------+

method 2:
+--------------------------------------------------+
|---------------+             +--------------------|
||local dataflow+------------>+tensorflow pipeline||
|---------------+             +--------------------|
+--------------------------------------------------+

'''

# write your model code
class LeNetModel(ModelDesc):
  def __init__(self, model_data_source=None):
    super(LeNetModel, self).__init__(LeNetModel, model_data_source=model_data_source)

  
  '''
  if you want to adopt method 3 to connect datasource to tensorflow, you have to comment model_input
  '''
  def model_input(self, is_training, *args, **kwargs):
    # online augmentation and preprocess module (for tensorflow)
    # for method 1 and method 2, all data process operations must be finished inner tensorflow
    if is_training:
      # for train stage, get both data and label info
      data, label = self.data_source.model_fn()

      # step 1: online augmentation

      # step 2: preprocess
      # use ctx.params.input_size to resize data to fit your model
      # ctx.params.input_size is set on LeNet_param.yaml

      # step 3: warp to batch
      # ctx.params.batch_size is set on LeNet_param.yaml
      # ctx.params.num_clones is set on LeNet_param.yaml, which is computing hardware number
      batch_data, batch_label = tf.train.shuffle_batch([data, label],
                                                       batch_size=ctx.params.batch_size,
                                                       num_threads=4,
                                                       capacity=200,
                                                       min_after_dequeue=100)
      data_queue = slim.prefetch_queue.prefetch_queue([batch_data, batch_label],
                                                      capacity=ctx.params.num_clones*2,
                                                      num_threads=ctx.params.num_clones*2)
      return data_queue
    else:
      # for non-train stage, only get data info
      data = self.data_source.model_fn()

      # step 1: preprocess
      # ctx.params.batch_size is set on LeNet_param.yaml
      batch_data = tf.train.batch([data],
                                  batch_size=ctx.params.batch_size,
                                  allow_smaller_final_batch=True)
      return batch_data
  

  def model_fn(self, is_training=True, *args, **kwargs):
    # write your own model code
    
    # for tensorflow
    # step 1: unwarp data
    batch_data = None
    batch_label = None
    if len(args) > 0:
      # for method 2
      # on train or test stage, unwarp data from args (which comes from model_input())
      if is_training:
        batch_data, batch_label = args[0].dequeue()
      else:
        batch_data = args[0]
    else:
      # for method 1
      # use placeholder
      batch_data = tf.placeholder(tf.uint8,
                                  shape=[ctx.params.batch_size, ctx.params.input_size, ctx.params.input_size, 1],
                                  name='data_node')

      if not is_training:
        batch_label = tf.placeholder(tf.int32,
                                     shape=[ctx.params.batch_size],
                                     name='label_node')

    # 转换数据类型
    batch_data = tf.cast(batch_data, tf.float32)

    # step 2: building model
    # 实现LeNet-5卷积神经网络
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=None,
                        weights_initializer=slim.variance_scaling_initializer()):

        # 卷积层：输入Tensor大小: batch x 28 x 28 x 1； 输出Tensor大小: batch x 24 x 24 x 6
        conv_1 = slim.conv2d(batch_data, 6, [5, 5], stride=1, activation_fn=tf.nn.relu, padding='VALID')
        print(conv_1)

        # 池化层：输入Tensor大小：batch x 24 x 24 x 6；输出Tensor大小：batch x 12 x 12 x 6
        pool_1 = slim.max_pool2d(conv_1, [2, 2], stride=2, padding='VALID')

        # 卷积层：输入Tensor大小：batch x 12 x 12 x 16；输出Tensor大小：batch x 8 x 8 x 16
        conv_2 = slim.conv2d(pool_1, 16, [5, 5], stride=1, activation_fn=tf.nn.relu, padding='VALID')
        print(conv_2)

        # 池化层：输入Tensor大小：batch x 8 x 8 x 16；输出Tensor大小：batch x 4 x 4 x 16
        pool_2 = slim.max_pool2d(conv_2, [2, 2], stride=2, padding='VALID')

        # 展开成一维Tensor，输入Tensor大小：batch x 4 x 4 x 16，输出Tensor大小：batch x 256
        fc_1 = tf.contrib.layers.flatten(pool_2)

        # 全连接层：输入Tensor大小：batch x 256；输出Tensor大小：batch x 120
        fc_1 = slim.fully_connected(fc_1, 120)

        # Relu激活层
        fc_1 = tf.nn.relu(fc_1)

        # 全连接层：输入Tensor大小：batch x 120；输出Tensor大小：batch x 84
        fc_2 = slim.fully_connected(fc_1, 84)
        # Relu激活层
        fc_2 = tf.nn.relu(fc_2)

        # 全连接层：输入Tensor大小：batch x 84；输出Tensor大小：batch x 10 (MIMIST 数据集总共10个类别)
        logits = slim.fully_connected(fc_2, 10)

        # step 3: output
        if is_training:
            # use logits to compute loss
            # 使用Logits计算交叉熵损失
            batch_label_one_hot = slim.one_hot_encoding(batch_label, 10)
            loss = tf.losses.softmax_cross_entropy(batch_label_one_hot, logits)
            return loss
        else:
            # use logits to compute model predict
            # 使用Logits计算分类概率
            predict = tf.nn.softmax(logits)
            return predict


##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  
  # 0.step bind data source and tensorflow data pipeline
  # method 2
  def _extract_func(*args, **kwargs):
      # Mnist dataset Antgo 解析并返回格式如下
      # img, {'id': id, 'category_id': label}
      # 送入模型需要完整的裸数据：img,label
      image = args[0][0]
      label = args[0][1]['category_id']

      return image, label

  data_source_size = data_source.size
  data_source = Node('extract', _extract_func, Node.inputs(data_source))

  data_source = TFQueueDataset(data_source,
                               [tf.uint8,
                                tf.int32],
                               [(ctx.params.input_size, ctx.params.input_size, 1),
                                ()])
  data_source.size = data_source_size

  # 1.step build model and trainer
  model = LeNetModel(data_source)

  trainer = TFTrainer(dump_dir, is_training=True)
  trainer.deploy(model)

  # method 1
  # # if you want to adopt 'method 3' feed data into model, you have to preprocess data by antgo provided methods
  # # preprocess data
  # # ...
  # # warp data to batch
  # batch_data_source = BatchData(Node.inputs(data_source), ctx.params.batch_size)
  #
  # iter = 0
  # for epoch in range(ctx.params.max_epochs):
  #   while True:
  #     try:
  #       input_name = ''         # input_name is your defined input name in your model
  #       input_binded_index = 0  # input_binded_index is binded index in your data source
  #
  #       label_name = ''         # label_name is your defined groundtruth name in your model
  #       label_binded_index = 1  # label_binded_index is binded index in your data source
  #       _, loss_val = trainer.run(batch_data_source.iterator_value(),
  #                                 binds={input_name:input_binded_index, label_name:label_binded_index})
  #
  #       if trainer.iter_at % 100 == 0:
  #         # record training info
  #         loss_channel.send(trainer.iter_at, loss_val)
  #
  #       # increment 1
  #       iter = iter + 1
  #     except:
  #       break

  # ctx.params.batch_size is set on LeNet_param.yaml
  # ctx.params.num_clones is set on LeNet_param.yaml, which is computing hardware number
  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size * ctx.params.num_clones))
    for _ in range(rounds):
      _, loss_val = trainer.run()
      if trainer.iter_at % 100 == 0:
          # record training info
          loss_channel.update(trainer.iter_at, loss_val)

    # save every epoch
    trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  
  # 0.step bind data source and tensorflow data pipeline
  # method 2
  data_source = TFQueueDataset(data_source,
                                 [tf.uint8],
                                 [(ctx.params.input_size, ctx.params.input_size, 1)])

  # 1.step build model and trainer
  model = LeNetModel(data_source)

  trainer = TFTrainer(dump_dir, is_training=False)
  trainer.deploy(model)

  # method 1
  # # if you want to adopt 'method 3' feed data into model, you have to preprocess data by antgo provided methods
  # # preprocess data
  # # ...
  # # warp data to batch
  # batch_data_source = BatchData(Node.inputs(data_source), ctx.params.batch_size)
  # for data in batch_data_source.iterator_value():
  #   # execute infer process
  #   input_data = data   # data from data source
  #   input_name = ''     # input_name is your defined input name in your model
  #
  #   # run model inference
  #   predict = trainer.run(**{input_name: input_data})
  #
  #   # post process predict result
  #   # ...
  #   post_process = predict
  #
  #   # record
  #   ctx.recorder.record({'RESULT': post_process})
  #
  while True:
    # run model inference
    predict = trainer.run()
    # post process predict result
    # ...
    post_process = predict

    # record
    ctx.recorder.record({'RESULT': post_process})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback


###################################################
###########    6.step test run         ############
###########                            ############
###################################################
if __name__ == '__main__':
  # 1.step debug training process
  debug_training_process(lambda :(np.random.randint(0, 255, (28, 28, 1)).astype(np.uint8),
                                  {'category_id': np.floor(np.random.random()*10).astype(np.int32)}),
                         param_config='LeNet_param.yaml')

  # 2.step debug infer process
  debug_infer_process(lambda : (np.random.randint(0, 255, (28, 28, 1)).astype(np.uint8)),
                      param_config='LeNet_param.yaml')
