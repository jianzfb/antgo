# -*- coding: UTF-8 -*-
# @Time    : 2019-06-06
# @File    : FCN_main.py
# @Author  : j
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
import vgg

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
loss_channel = ctx.dashboard.create_channel("loss", "NUMERIC")
# bind channel to chart,every chart could include some channels
ctx.dashboard.create_chart([loss_channel], "Loss Curve", "step", "value")


##################################################
######## 2.step custom model building code  ######
##################################################


##################################################
######## 2.1.step custom dataset parse code ######
########    write dataset parse code if     ######
#####  you want to parse your private dataset ####
##################################################
class FCNDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(FCNDataset, self).__init__(train_or_test, dir, params)
    # 由于使用Tensorflow数据读取机制，必须手工指定数据集大小
    self.dataset_size = 1464+10582 if train_or_test == 'train' else 1449

  @property
  def size(self):
    # 返回数据集大小
    return self.dataset_size

  def model_fn(self, *args, **kwargs):

    # 设置tfrecord记录关键词 key 以及解析格式
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'mask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'mask/format': tf.FixedLenFeature((), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'mask': slim.tfexample_decoder.Image('mask/encoded', 'mask/format', channels=1),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # 设置如何过滤tfrecords，以及创建数据集对象
    tfrecord_pattern = '*.tfrecord'
    dataset = slim.dataset.Dataset(
                data_sources=os.path.join(self.dir, self.train_or_test, tfrecord_pattern),
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=self.size,
                items_to_descriptions=None)

    # 设置数据集读取对象
    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=12,
            common_queue_capacity=3000,
            common_queue_min=1000,
            shuffle=True if self.train_or_test else False,
            num_epochs=None)

        image, label = provider.get(['image', 'mask'])
        label = tf.squeeze(label, -1)
        return image, label


##################################################
######## 2.2.step custom metric code        ######
##################################################
'''
class FCNMeasure(AntMeasure):
  def __init__(self, task):
    super(FCNMeasure, self).__init__(task, 'FCNMeasure')
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
there are three ways to feed data into tensorflow framework.
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
class FCNModel(ModelDesc):
  def __init__(self, model_data_source=None):
    super(FCNModel, self).__init__(FCNModel, model_data_source=model_data_source)

  
  '''
  if you want to adopt method 1 to connect datasource to tensorflow, you have to comment model_input
  '''
  def model_input(self, is_training, *args, **kwargs):
    # online augmentation and preprocess module (for tensorflow)
    # for method 2 and method 3, all data process operations must be finished in tensorflow
    if is_training:
      # for train stage, get both data and label info
      data, label = self.data_source.model_fn()

      # step 1: online augmentation
      # random rotation
      # degree_angle = 30 * tf.random_uniform((), -1, 1)
      # radian = degree_angle * np.pi / 180
      # data = tf.contrib.image.rotate(tf.expand_dims(data, 0), radian)
      # data = tf.squeeze(data, 0)
      data = tf.image.random_brightness(data, 0.3)
      data = tf.image.random_contrast(data, 0.1, 0.6)
      data = tf.image.random_saturation(data, 0.1, 0.6)

      # step 2: preprocess
      # use ctx.params.input_size to resize data to fit your model
      # ctx.params.input_size is set on FCN_param.yaml
      # 预处理第一步：resize 到模型输入大小
      data = tf.image.resize_images(data, size=(ctx.params.input_size, ctx.params.input_size))
      label = tf.image.resize_images(tf.expand_dims(label, -1),
                                     size=(ctx.params.input_size, ctx.params.input_size),
                                     method=1)
      label = tf.squeeze(label, -1)

      # 预处理第二步：减均值
      rgb_channels = tf.split(data, 3, 2)
      rgb_channels[0] = rgb_channels[0] - 128.0
      rgb_channels[1] = rgb_channels[1] - 128.0
      rgb_channels[2] = rgb_channels[2] - 128.0

      # 预处理第三步：除方差
      rgb_channels[0] = rgb_channels[0] / 255.0
      rgb_channels[1] = rgb_channels[1] / 255.0
      rgb_channels[2] = rgb_channels[2] / 255.0

      data = tf.concat(rgb_channels, -1)

      # step 3: warp to batch
      # ctx.params.batch_size is set on FCN_param.yaml
      # ctx.params.num_clones is set on FCN_param.yaml, which is computing hardware number
      batch_data, batch_label = tf.train.shuffle_batch([data, label],
                                                       batch_size=ctx.params.batch_size,
                                                       num_threads=4,
                                                       capacity=4000,
                                                       min_after_dequeue=2000)
      data_queue = slim.prefetch_queue.prefetch_queue([batch_data, batch_label],
                                                      capacity=ctx.params.num_clones*2,
                                                      num_threads=ctx.params.num_clones*2)
      return data_queue
    else:
      # for non-train stage, only get data info
      data = self.data_source.model_fn()
      origianl_img = data
      shape = tf.shape(data)

      # step 1: preprocess
      # use ctx.params.input_size to resize data to fit your model
      # ctx.params.input_size is set on FCN_param.yaml
      # 预处理第一步：resize 到模型输入大小
      data = tf.image.resize_images(data, size=(ctx.params.input_size, ctx.params.input_size))

      # 预处理第二步：减均值
      rgb_channels = tf.split(data, 3, 2)
      rgb_channels[0] = rgb_channels[0] - 128.0
      rgb_channels[1] = rgb_channels[1] - 128.0
      rgb_channels[2] = rgb_channels[2] - 128.0

      # 预处理第三步：除方差
      rgb_channels[0] = rgb_channels[0] / 255.0
      rgb_channels[1] = rgb_channels[1] / 255.0
      rgb_channels[2] = rgb_channels[2] / 255.0

      data = tf.concat(rgb_channels, -1)

      # step 2: warp to batch
      batch_data = tf.expand_dims(data, 0)
      return batch_data, shape, origianl_img
  

  def model_fn(self, is_training=True, *args, **kwargs):
    # write your own model code
    
    # for tensorflow
    # step 1: unwarp data
    batch_data = None
    batch_label = None
    origianl_img = None
    shape = None
    if len(args) > 0:
      # for method 2 and method 3
      # on train or test stage, unwarp data from args (which comes from model_input())
      if is_training:
        batch_data, batch_label = args[0].dequeue()
      else:
        batch_data, shape, origianl_img = args[0]
    else:
      # for method 1
      # use placeholder
      batch_data = tf.placeholder(tf.float32,
                                  shape=[ctx.params.batch_size, ctx.params.input_size, ctx.params.input_size, 3],
                                  name='data_node')

      if not is_training:
        batch_label = tf.placeholder(tf.int32,
                                     shape=[ctx.params.batch_size, ctx.params.input_size, ctx.params.input_size],
                                     name='label_node')

    # step 2: building model
    # 使用VGG16作为基础网络
    with slim.arg_scope(vgg.vgg_arg_scope()):
      net, end_points = vgg.vgg_16(batch_data, None, is_training)

    # 抽取pool5, pool4, pool3 中间层
    vgg_pool5 = end_points['vgg_16/pool5']
    vgg_pool4 = end_points['vgg_16/pool4']
    vgg_pool3 = end_points['vgg_16/pool3']

    with tf.variable_scope(None, 'FCN', [vgg_pool5, vgg_pool4, vgg_pool3]):
      conv6 = slim.conv2d(vgg_pool5, 4096, [7, 7], padding='SAME', activation_fn=tf.nn.relu)
      conv7 = slim.conv2d(conv6, 4096, [1, 1], padding='SAME', activation_fn=tf.nn.relu)
      conv7_dropout = slim.dropout(conv7, is_training=is_training)

      # FCN32S
      # 使用最后一层（分辨率16x16）进行语义类别预测
      score_32 = slim.conv2d(conv7_dropout, ctx.params.class_num, [1, 1], padding='SAME')
      score_32_up = slim.convolution2d_transpose(score_32, vgg_pool4.get_shape()[-1], [4, 4], [2, 2])

      # FCN16S
      # 结合来自pool4的特征图（分辨率32x32）进行进一步语义类别预测
      score_16 = score_32_up + vgg_pool4
      score_16_up = slim.convolution2d_transpose(score_16, vgg_pool3.get_shape()[-1], [4, 4], [2, 2])

      # FCN8S
      # 集合来自pool3的特征图（分辨率64x64）进行进一步语义类别预测
      score_8 = score_16_up + vgg_pool3
      score_8_up = slim.convolution2d_transpose(score_8, ctx.params.class_num, [16, 16], [8, 8], activation_fn=None)

      if is_training:
        one_hot_batch_label = tf.one_hot(batch_label, ctx.params.class_num)
        # cross entropy
        fcn8_loss = tf.losses.softmax_cross_entropy(one_hot_batch_label, score_8_up)
        return fcn8_loss
      else:
        logits = tf.nn.softmax(score_8_up)
        logits = tf.image.resize_images(logits, (shape[0], shape[1]))
        return logits, origianl_img


##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  
  # 0.step bind data source and tensorflow data pipeline
  # method 2
  if data_source is not None and not data_source.name.startswith('FCNDataset'):
    data_source = TFQueueDataset(data_source,
                                 [tf.uint8,
                                  tf.int32],
                                 [(None, None, 3),
                                  (None, None)])

  # 1.step build model and trainer
  model = FCNModel(data_source)

  trainer = TFTrainer(dump_dir, is_training=True)
  trainer.deploy(model)

  # method 1
  # # if you want to adopt 'method 1' feed data into model, you have to preprocess data by antgo provided methods
  # # preprocess data
  # # ...
  # # warp data to batch
  # batch_data_source = BatchData(Node.inputs(data_source), ctx.params.batch_size)
  #
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
  #     except:
  #       break

  # ctx.params.batch_size is set on FCN_param.yaml
  # ctx.params.num_clones is set on FCN_param.yaml, which is computing hardware number
  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size * ctx.params.num_clones))
    for _ in range(rounds):
      _, loss_val = trainer.run()
      if trainer.iter_at % 100 == 0:
        # record training info
        loss_channel.upate(trainer.iter_at, loss_val)

    # save every epoch
    if epoch % 2 == 0:
        trainer.snapshot(epoch)
  


###################################################
######## 4.step define infer process     ##########
###################################################
import cv2
def infer_callback(data_source, dump_dir):
  
  # 0.step bind data source and tensorflow data pipeline
  # method 3
  # use customized FCNDataset to parse dataset completely

  # method 2
  if data_source is not None and not data_source.name.startswith('FCNDataset'):
    data_source = TFQueueDataset(data_source,
                                 [tf.uint8],
                                 [(None, None, 3)])

  # 1.step build model and trainer
  # only support batch size = 1
  ctx.params.batch_size = 1

  model = FCNModel(data_source)
  trainer = TFTrainer(dump_dir, is_training=False)
  trainer.deploy(model)

  # method 1
  # # if you want to adopt 'method 1' feed data into model, you have to preprocess data by antgo provided methods
  # # preprocess data
  # # ...
  # # warp data to batch
  # batch_data_source = BatchData(Node.inputs(data_source), ctx.params.batch_size)
  # while True:
  #   try:
  #     input_name = ''         # input_name is your defined input name in your model
  #     input_binded_index = 0  # input_binded_index is binded index in your data source
  #
  #     label_name = ''         # label_name is your defined groundtruth name in your model
  #     label_binded_index = 1  # label_binded_index is binded index in your data source
  #     predict = trainer.run(batch_data_source.iterator_value(),
  #                               binds={input_name:input_binded_index, label_name:label_binded_index})
  #   except:
  #     break

  rounds = data_source.size // ctx.params.batch_size
  for _ in range(rounds):
    # run model inference
    predict, image = trainer.run()
    # post process predict result
    predict = np.argmax(predict, -1)
    predict = np.split(predict, ctx.params.batch_size, 0)

    cv2.imshow('a', image.astype(np.uint8))
    cv2.imshow('b', ((np.squeeze(predict[0], 0)/21.0)*255).astype(np.uint8))
    cv2.waitKey()

    # record
    ctx.recorder.record([{'RESULT': np.squeeze(result, 0).astype(np.int32),
                          'RESULT_TYPE': 'IMAGE'} for result in predict])

  

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
  # # 1.step debug training process
  # debug_training_process(lambda :(np.random.randint(0,255,(512,512,3)).astype(np.uint8),
  #                                 np.random.randint(0,10, (512,512))).astype(np.int32),
  #                        param_config='FCN_param.yaml')

  # 2.step debug infer process
  debug_infer_process(lambda : (np.random.randint(0,255,(512,512,3)).astype(np.uint8)),
                      param_config='FCN_param.yaml')