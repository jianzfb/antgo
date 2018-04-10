# -*- coding: UTF-8 -*-
# @Time    : 18-03-19
# @File    : ali_fashionai_landmark.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from antgo.codebook.tf.layers import *
import tensorflow.contrib.slim as slim
from antgo.trainer.tftrainer import *
from antgo.dataflow.imgaug.regular import *
from antgo.codebook.tf.dataset import *
from antgo.codebook.tf.preprocess import *
from antgo.utils._resize import resize
import densenet
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers
##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2.step model building (tensorflow) ######
##################################################
def gcn_module(inputs,
               num_class,
               kernel_size,
               scope=None):
  with slim.variable_scope.variable_scope(scope, 'gcn_module', [inputs]):
    with slim.arg_scope([layers_lib.conv2d],
        padding='SAME',
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
      left_conv1 = layers_lib.conv2d(inputs, num_class, [kernel_size, 1])
      left_conv2 = layers_lib.conv2d(left_conv1, num_class, [1, kernel_size])
      
      right_conv1 = layers_lib.conv2d(inputs, num_class, [1, kernel_size])
      right_conv2 = layers_lib.conv2d(right_conv1, num_class, [kernel_size, 1])
      
      result_sum = tf.add(left_conv2, right_conv2, name='gcn_module')
      return result_sum

def gcn_br(inputs, scope):
  with slim.variable_scope.variable_scope(scope, 'gcn_br', [inputs]):
    with slim.arg_scope([layers_lib.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=initializers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                        biases_initializer=init_ops.zeros_initializer(),
                        biases_regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
      num_class = inputs.get_shape()[3]
      conv = layers_lib.conv2d(inputs, num_class, [3, 3])
      conv = layers_lib.conv2d(conv, num_class, [3, 3], activation_fn=None)
      result_sum = tf.add(inputs, conv, name='fcn_br')
      return result_sum

class HourglassModel(ModelDesc):
  def __init__(self):
    super(HourglassModel, self).__init__()
    self._nstack = ctx.params.nstacks
    self._nfeat = ctx.params.nfeat
    self._nlow = ctx.params.nlow
    self._output_dim = ctx.params.output_dim # 16
    self._dropout_rate = ctx.params.dropout_rate
  
  def arg_scope(self, is_training):
    # branch side
    batch_norm_params = {
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=None,
                        biases_initializer=None,
                        weights_initializer=slim.variance_scaling_initializer(),
                        ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
            return scope

  def _residual(self, inputs, out_channels, name='residual_block'):
    with tf.variable_scope(name):
      # three convs
      norm_1 = slim.batch_norm(inputs, activation_fn=tf.nn.relu)
      conv_1 = slim.conv2d(norm_1, out_channels/2, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      norm_2 = slim.batch_norm(conv_1, activation_fn=tf.nn.relu)
      conv_2 =slim.conv2d(norm_2, out_channels/2, [3, 3], stride=1, normalizer_fn=None, activation_fn=None)
      
      norm_3 = slim.batch_norm(conv_2, activation_fn=tf.nn.relu)
      conv_3 = slim.conv2d(norm_3, out_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      # skip
      res_layer = None
      if inputs.get_shape().as_list()[3] == out_channels:
        res_layer = inputs
      else:
        res_layer = slim.conv2d(inputs, out_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      return tf.add(res_layer, conv_3)
  
  def _hourglass(self, inputs, n, num_out, name='hourglass'):
    with tf.variable_scope(name):
      # upper branch
      up_1 = self._residual(inputs, num_out, name='up_1')
      # lower branch
      low_ = slim.max_pool2d(inputs, [2, 2], stride=2)
      low_1 = self._residual(low_, num_out, name='low_1')
      
      if n > 0:
        low_2 = self._hourglass(low_1, n-1, num_out, name='low_2')
      else:
        low_2 = self._residual(low_1, num_out, name='low_2')
      
      low_3 = self._residual(low_2, num_out, name='low_3')
      up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name='upsampling')
      return tf.nn.relu(tf.add(up_2, up_1), name='out_hg')
  
  def model_input(self, is_training, data_source):
    if not is_training:
      return None
    
    warp_data = Node('warp', _data_warp, Node.inputs(data_source))
    reorganized_data = Node('reorganize', _data_reorganize, Node.inputs(warp_data))
  
    qq = DatasetQueue(reorganized_data,
                      [tf.float32, tf.float32, tf.int32, tf.int32],
                      [(ctx.params.input_size, ctx.params.input_size, 4),
                       (ctx.params.nstacks, ctx.params.hm_size, ctx.params.hm_size, len(ctx.params.joints)),
                       (24,),
                       (ctx.params.input_size, ctx.params.input_size, 1)])
    image, gtmap, pick_index, attention_region = qq.dequeue()

    random_angle = (tf.to_float(tf.random_uniform([1]))[0] * 2 - 1) * 10.0 / 180.0 * 3.14
    # for image
    image = tf.expand_dims(image, 0)
    rotated_image = tf.contrib.image.rotate(image, random_angle)
    rotated_image = tf.squeeze(rotated_image, 0)

    # for gtmap
    rotated_gtmap = tf.contrib.image.rotate(gtmap, random_angle)
    
    # for attention
    rotated_attention_region = tf.contrib.image.rotate(attention_region, random_angle)
    
    # make batch
    rotated_images, rotated_gtmaps, pick_indexs, attention_regions = tf.train.batch([rotated_image, rotated_gtmap, pick_index, rotated_attention_region], ctx.params.batch_size)
    data_queue = slim.prefetch_queue.prefetch_queue([rotated_images, rotated_gtmaps, pick_indexs, attention_regions])
    return data_queue
  
  def model_fn(self, is_training=True, *args, **kwargs):
    #inputs: batch_size x 256 x 256 x 3
    #labels: batch_size x nstack x 64 x 64 x output_dim(landmark number)
    inputs = None
    labels = None
    
    if is_training:
      inputs, labels, pick_indexs, attention_regions = args[0].dequeue()
    else:
      inputs = tf.placeholder(tf.float32, shape=(ctx.params.input_size,ctx.params.input_size, 4), name='input_x')
      inputs = tf.expand_dims(inputs, 0)
    
    with tf.variable_scope('attention'):
      # stage - 1 dense block
      input_r, input_g, input_b,input_a = tf.split(inputs, 4, axis=3)
      input_rgb = tf.concat((input_r,input_g,input_b), -1)
      input_rgb_128 = tf.image.resize_bilinear(input_rgb, (128, 128))
      with slim.arg_scope(densenet.densenet_arg_scope(weight_decay=2e-5)):
        _, each_layer_output = densenet.densenet(input_rgb_128,
                                                 num_classes=2,
                                                 reduction=0.5,
                                                 growth_rate=32,
                                                 num_filters=64,
                                                 num_layers=[6],
                                                 is_training=is_training,
                                                 reuse=None,
                                                 scope='densenet121')

      # stage - 2 gcn block
      with tf.variable_scope(None, 'GCN', [each_layer_output], reuse=None):
        # 64 x 64
        res_1 = each_layer_output['densenet121/root_block']
        # 32 x 32
        res_2 = each_layer_output['attention/densenet121/dense_block1']

        res_1_gcn = gcn_module(res_1, 2, 15, 'gcn_1')
        res_2_gcn = gcn_module(res_2, 2, 15, 'gcn_2')

        res_2_gcn = gcn_br(res_2_gcn, 'br_2')
        _, height, width, _ = res_2_gcn.shape
        res_2_upsample = tf.image.resize_bilinear(res_2_gcn, (int(height)*2, int(width)*2))
        res_1_gcn = gcn_br(res_2_upsample + res_1_gcn, 'br_1')
        attention_logits = res_1_gcn
        attention_prob = tf.nn.softmax(attention_logits)
    
    with slim.arg_scope(self.arg_scope(is_training)):
      with tf.variable_scope('model'):
        with tf.variable_scope('preprocessing'):
          # inputs batch_size x 256 x 256 x 4
          pad_1 = tf.pad(inputs, np.array([[0, 0], [2, 2], [2, 2], [0, 0]]))
          conv_1 = slim.conv2d(pad_1, 64, [6, 6], stride=2, scope='256to128', padding='VALID')
          res_1 = self._residual(conv_1, 128, name='r1')
          pool_1 = tf.contrib.layers.max_pool2d(res_1, [2, 2], [2, 2], padding='VALID')
          res_2 = self._residual(pool_1, 128, name='r2')
          res_3 = self._residual(res_2, ctx.params.nfeat, name='r3')

        hg = [None] * self._nstack
        ll = [None] * self._nstack
        ll_ = [None] * self._nstack
        drop = [None] * self._nstack
        out = [None] * self._nstack
        out_ = [None] * self._nstack
        sum_ = [None] * self._nstack
        with tf.variable_scope('stacks'):
          with tf.variable_scope('stage_0'):
            hg[0] = self._hourglass(tf.concat((res_3, attention_prob), -1), self._nlow, self._nfeat, 'hourglass')
            drop[0] = slim.dropout(hg[0], keep_prob=self._dropout_rate, is_training=is_training)
            ll[0] = slim.conv2d(drop[0], self._nfeat, [1, 1], stride=1, scope='cc')
            ll_[0] = slim.conv2d(ll[0], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='ll')
            
            out[0] = slim.conv2d(ll[0], self._output_dim, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
            out_[0] = slim.conv2d(out[0], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out_')
            sum_[0] = tf.add_n([out_[0], res_3, ll_[0]], name='merge')
            
          for i in range(1, self._nstack - 1):
            with tf.variable_scope('stage_'+str(i)):
              hg[i] = self._hourglass(tf.concat((sum_[i-1], attention_prob), -1), self._nlow, self._nfeat, 'hourglass')
              drop[i] = slim.dropout(hg[i], keep_prob=self._dropout_rate, is_training=is_training)
              ll[i] = slim.conv2d(drop[i], self._nfeat, [1, 1], stride=1, scope='cc')
              ll_[i] = slim.conv2d(ll[i], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='ll')
              
              out[i] = slim.conv2d(ll[i], self._output_dim, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
              out_[i] = slim.conv2d(out[i], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out_')
              sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[i]], name='merge')
              
          with tf.variable_scope('stage_'+str(self._nstack - 1)):
            hg[self._nstack - 1] = self._hourglass(tf.concat((sum_[self._nstack - 2], attention_prob), -1), self._nlow, self._nfeat, 'hourglass')
            drop[self._nstack - 1] = slim.dropout(hg[self._nstack - 1], keep_prob=self._dropout_rate, is_training=is_training)
            ll[self._nstack - 1] = slim.conv2d(drop[self._nstack - 1], self._nfeat, [1, 1], stride=1, scope='cc')
            
            out[self._nstack - 1] = slim.conv2d(ll[self._nstack - 1], self._output_dim, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
            
          output = tf.stack(out, axis=1, name='final_output')

          if is_training:
            # loss 1: attention region loss
            attention_regions_one_hot = slim.one_hot_encoding(tf.squeeze(attention_regions, -1), 2)
            attention_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.image.resize_bilinear(attention_logits,
                                                                                                     (ctx.params.input_size,
                                                                                                      ctx.params.input_size)),
                                                                      labels=attention_regions_one_hot)
            attention_loss = tf.reduce_mean(attention_loss)
            tf.losses.add_loss(0.5 * attention_loss)

            # loss 2: joint loss
            # shape batch_size x nstack x 64 x 64 x 24
            loss_mat = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels)
            
            # focal loss (computing weight of every sample)
            # shape: batch_size x nstacks x 64 x 64 x 24
            sigmoid_logits = tf.nn.sigmoid(output)
            # hard or easy
            hard_weight = tf.pow(1 - (sigmoid_logits * labels + (1-sigmoid_logits) * (1-labels)), 2)
            sample_weight = hard_weight * 50.0
            
            # weight loss
            loss_mat = loss_mat * sample_weight
            
            # attention on category joints (local attention)
            loss_mat_reshpae = tf.transpose(loss_mat, perm=(0, 4, 1, 2, 3))
            loss_mat_reshpae = tf.reshape(loss_mat_reshpae, (-1, ctx.params.nstacks, ctx.params.hm_size, ctx.params.hm_size))
            
            useful_index = tf.reshape(pick_indexs, (-1,))
            useful_index = tf.where(tf.equal(useful_index, 1))
            useful_index = tf.squeeze(useful_index, 1)
            loss_mat_reshpae = tf.gather(loss_mat_reshpae, useful_index)
            
            #
            loss = tf.reduce_mean(loss_mat_reshpae)
            tf.losses.add_loss(loss)
            return loss, attention_loss
          else:
            output = tf.nn.sigmoid(output)
            return output
##################################################
######## 3.step define training process  #########
##################################################
def _makeGaussian(height, width, sigma=3, center=None):
  """ Make a square gaussian kernel.
  size is the length of a side of the square
  sigma is full-width-half-maximum, which
  can be thought of as an effective radius.
  """
  x = np.arange(0, width, 1, float)
  y = np.arange(0, height, 1, float)[:, np.newaxis]
  if center is None:
    x0 = width // 2
    y0 = height // 2
  else:
    x0 = center[0]
    y0 = center[1]
  return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def _generate_hm(height, width, joints, maxlength, weight):
  num_joints = joints.shape[0]
  hm = np.zeros((height, width, num_joints), dtype=np.float32)
  for i in range(num_joints):
    if joints[i, 0] > 0 and joints[i, 1] > 0 and weight[i] == 1:
      s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
      hm[:, :, i] = _makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
    else:
      hm[:, :, i] = np.zeros((height, width))
  return hm


def _data_warp(*args, **kwargs):
  data, label = args[0]
  image, _ = data  # ignore category
  return image, label


def _data_reorganize(*args, **kwargs):
  train_img = args[0][0]  # 64 x 64 x 4
  train_img_annotation = args[0][1]  # {}
  
  category_dict = {'blouse': 0,
                   'dress': 1,
                   'outwear': 2,
                   'skirt': 3,
                   'trousers': 4}
  
  train_img = resize(train_img, [ctx.params.input_size, ctx.params.input_size])
  train_img = train_img / 255.0
  prior = category_dict[train_img_annotation['category']] * np.ones((ctx.params.input_size, ctx.params.input_size, 1))
  prior = prior / float(len(category_dict))
  train_img = np.concatenate((train_img, prior), axis=-1)
  
  landmarks = train_img_annotation['landmark']
  joints = np.zeros((len(ctx.params.joints), 2))
  ##################################################
  visibles = -1 * np.ones((len(ctx.params.joints)))
  index_pick = np.zeros((24), dtype=np.int32)
  
  x_list = []
  y_list = []
  info = train_img_annotation['info']
  old_height, old_width = info[0:2]

  for joint_index, joint_global_index, x, y, visible in landmarks:
    joints[joint_global_index, 0] = x
    joints[joint_global_index, 1] = y
    
    visibles[joint_global_index] = visible
    if visible == 1 or visible == 0:
      index_pick[joint_global_index] = 1
      x_list.append(x/float(old_width) * ctx.params.input_size)
      y_list.append(y/float(old_height) * ctx.params.input_size)
      
  x_min = int(np.min(x_list))
  x_max = int(np.max(x_list))
  y_min = int(np.min(y_list))
  y_max = int(np.max(y_list))
  
  height, width = train_img.shape[0:2]
  attention_region = np.zeros((height, width, 1), dtype=np.int32)
  attention_region[y_min:y_max, x_min:x_max, 0] = 1
  
  joints = np.array(joints).reshape((-1, 2))
  joints[:, 0] = joints[:, 0] / float(old_width) * ctx.params.hm_size
  joints[:, 1] = joints[:, 1] / float(old_height) * ctx.params.hm_size
  
  train_weight = np.zeros((len(ctx.params.joints)), np.float32)
  train_weight[np.where(visibles == 1)] = 1
  train_weight[np.where(visibles == 0)] = 1
  
  hm = _generate_hm(ctx.params.hm_size, ctx.params.hm_size, joints, ctx.params.hm_size, train_weight)
  hm = np.expand_dims(hm, axis=0)
  hm = np.repeat(hm, ctx.params.nstacks, axis=0)
  
  return train_img, hm, index_pick, attention_region


def training_callback(data_source, dump_dir):
  ##########  2.step build model     ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.deploy(HourglassModel())

  iter = 0
  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size * ctx.params.num_clones))
    for _ in range(rounds):
      _, loss_val, attention_loss = tf_trainer.run()
      if iter % 100 == 0:
        print('loss %f and attention loss %f'%(loss_val))
      iter = iter + 1
      
    # save
    tf_trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
import cv2
def _data_filter(*args, **kwargs):
  image, category, _ = args[0]

  category_dict = {'blouse': 0,
                   'dress': 1,
                   'outwear': 2,
                   'skirt': 3,
                   'trousers': 4}

  test_image = resize(image, [ctx.params.input_size, ctx.params.input_size])
  test_image = test_image / 255.0
  prior = category_dict[category] * np.ones((ctx.params.input_size, ctx.params.input_size, 1))
  prior = prior / float(len(category_dict))
  test_image = np.concatenate((test_image, prior), axis=-1)

  return test_image


def infer_callback(data_source, dump_dir):
  ##########  1.step data preprocess ##############
  warp_data = Node('warp', _data_filter, Node.inputs(data_source))
  
  ##########  2.step building model  ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=False)
  tf_trainer.deploy(HourglassModel())
  
  key_points = ['neckline_left',  # 左领部
                'neckline_right',  # 右领部
                'center_front',  # 中线
                'shoulder_left',  # 左肩部
                'shoulder_right',  # 右肩部
                'armpit_left',  # 左腋窝
                'armpit_right',  # 右腋窝
                'waistline_left',  # 左腰部
                'waistline_right',  # 右腰部
                'cuff_left_in',  # 左袖口内
                'cuff_left_out',  # 左袖口外
                'cuff_right_in',  # 右袖口内
                'cuff_right_out',  # 右袖口外
                'top_hem_left',  # 左衣摆
                'top_hem_right',  # 右衣摆
                'waistband_left',  # 左腰部
                'waistband_right',  # 右腰部
                'hemline_left',  # 左裙摆
                'hemline_right',  # 右裙摆
                'crotch',  # 裆部
                'bottom_left_in',  # 左裤脚内
                'bottom_left_out',  # 左裤脚外
                'bottom_right_in',  # 右裤脚内
                'bottom_right_out',  # 右裤脚外
                ]
  
  category_map = {'blouse': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, ],
                  'dress': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
                  'outwear': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                  'skirt': [15, 16, 17, 18, ],
                  'trousers': [15, 16, 19, 20, 21, 22, 23]}
  
  # submit file
  fp = open('result.csv', 'w')
  submit_table_head = ','.join(key_points)
  submit_table_head = 'image_id,image_category,' + submit_table_head + '\n'
  fp.write(submit_table_head)
  
  ##########  3.step predict sample  ##############
  generator = warp_data.iterator_value()
  while True:
    try:
      # predict landmark
      output, = tf_trainer.run(generator, binds={'input_x': 0})
      output = output[0][7]
      
      # original data
      predict_img, category, image_file = data_source.get_value()
      height, width = predict_img.shape[0:2]
      
      predict_joints = -1 * np.ones((len(ctx.params.joints), 3), dtype=np.uint32)
      
      submit_table_item = '%s,%s' % (image_file, category)
      for joint_i in range(len(ctx.params.joints)):
        if joint_i not in category_map[category]:
          submit_table_item = submit_table_item + ',-1_-1_-1'
          continue
        
        index = np.unravel_index(output[:, :, joint_i].argmax(), (ctx.params.hm_size, ctx.params.hm_size))
        if output[index[0], index[1], joint_i] > ctx.params.thresh:
          # resize to original scale
          y = index[0]
          y = y / float(ctx.params.hm_size) * height
          x = index[1]
          x = x / float(ctx.params.hm_size) * width
          predict_joints[joint_i, 0] = x
          predict_joints[joint_i, 1] = y
          predict_joints[joint_i, 2] = 1
          
          submit_table_item = submit_table_item + ',%d_%d_1' % (int(x), int(y))
          
          predict_img[int(y), int(x)] = 255
          # cv2.circle(predict_img, center=tuple((int(x), int(y))), radius=5, color=(255,0,0), thickness=-1)
          # cv2.putText(predict_img,ctx.params.joints[joint_i], (int(x+5),int(y+5)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
          #
        else:
          submit_table_item = submit_table_item + ',-1_-1_-1'
      
      submit_table_item = submit_table_item + '\n'
      
      # 4.step record predicted landmark
      ctx.recorder.record((predict_joints, category, image_file))
      
      # 5.step record to standard submit format
      fp.write(submit_table_item)
      
      # # # show
      # cv2.imshow('ss1', predict_img.astype(np.uint8))
      # cv2.waitKey()
    except:
      break
  
  fp.close()

###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback