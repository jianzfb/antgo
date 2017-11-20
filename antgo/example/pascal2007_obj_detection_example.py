# -*- coding: UTF-8 -*-
# @Time    : 17-11-17
# @File    : pascal2007_obj_detection_example.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from antgo.trainer.tftrainer import *
from nets.ssd import *
import tf_extended as tfe
from antgo.utils.helper import *
from antgo.utils._resize import *

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2 step define chart channel ###########
##################################################
# channel 1
loss_channel = ctx.job.create_channel('loss', 'NUMERIC')

# chart 1
ctx.job.create_chart([loss_channel], 'loss', 'step', 'value')


##################################################
######## 3.step model building (tensorflow) ######
##################################################
def tf_positive_and_negative_selecting_strategy(proposals, logits, gt_boxes, gt_labels, neg_pos_ratio, name):
  return tf.py_func(batch_positive_and_negative_selecting_strategy,
                    [proposals, logits, gt_boxes, gt_labels, neg_pos_ratio],
                    [tf.int32, tf.float32], name)


def smooth_l1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5

    condition = tf.less(tf.abs(x), 1.0)
    re = tf.where(condition, l2, l1)

    return re


class SSDObjDModel(ModelDesc):
  def __init__(self):
    super(SSDObjDModel, self).__init__()
  
  def model_fn(self, is_training=True, *args, **kwargs):
    images = tf.placeholder(shape=(1, 300, 300, 3), dtype=tf.float32, name='ssd-images')
    bboxes = tf.placeholder(shape=(1, None, 4), dtype=tf.float32, name='ssd-bbox')
    labels = tf.placeholder(shape=(1, None), dtype=tf.float32, name='ssd-labels')
    
    # extend label (background is always 1)
    ext_labels = labels
    ext_classes_num = self.num_classes + 1
    
    # standard ssd model
    with slim.arg_scope(ssd_arg_scope()):
      predictions, logits, localisations_delta = ssd(images, ext_classes_num, is_training)
    
    # localisations (np.narray)
    default_anchors = ssd_anchors((300, 300))
    localisations = ssd_bbox_decoder(default_anchors, localisations_delta)
    
    # shape: batch_size x N x N x anchors_num x 4 (x0,y0,x1,y1)
    # coordinate has been normalized (0 ~ 1)
    # localisations = ssd_common.tf_ssd_bboxes_decode(localisations_delta, default_anchors, is_yx=False)
    # multi_localisations = []
    # for l in localisations:
    #     l_shape = tfe.get_shape(l)
    #     l = tf.reshape(l, [l_shape[0], -1, 4])
    #     multi_localisations.append(l)
    # localisations = tf.concat(multi_localisations, axis=1)
    
    if is_training:
      # predict localisation
      localisations = tf.concat(localisations, axis=1)
      
      multi_logits = []
      for p in logits:
        p_shape = tfe.get_shape(p)
        p = tf.reshape(p, [p_shape[0], -1, ext_classes_num])
        multi_logits.append(p)
      logits = tf.concat(multi_logits, axis=1)
      
      multi_localisations_delta = []
      for l in localisations_delta:
        l_shape = tfe.get_shape(l)
        l = tf.reshape(l, [l_shape[0], -1, 4])
        multi_localisations_delta.append(l)
      localisations_delta = tf.concat(multi_localisations_delta, axis=1)
      
      # select positive and negative samples
      # proposal_labels shape (-1,-1)
      # proposal_targets shape (-1,-1,4)
      proposal_labels, proposal_targets = \
        tf_positive_and_negative_selecting_strategy(localisations, logits, bboxes, ext_labels, 3, 'matching_strategy')
      proposal_labels = tf.reshape(proposal_labels, [-1])
      proposal_targets = tf.reshape(proposal_targets, [-1, 4])
      
      # reorganized
      localisations_delta = tf.reshape(localisations_delta, [-1, 4])
      logits = tf.reshape(logits, [-1, ext_classes_num])
      
      # loss function
      # loss part 1 (cls loss)
      ssd_cls_logits = tf.reshape(tf.gather(logits, tf.where(tf.not_equal(proposal_labels, -1))),
        [-1, ext_classes_num])
      ssd_cls_label = tf.reshape(tf.gather(proposal_labels, tf.where(tf.not_equal(proposal_labels, -1))), [-1])
      ssd_cls_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ssd_cls_logits, labels=ssd_cls_label), 0,
        name='cls-loss')
      
      # loss part 2 (box regression loss)
      positive_boxes_index = tf.where(tf.greater(proposal_labels, 0), name='where_positive_boxes')
      ssd_positive_proposal_predicated = tf.reshape(tf.gather(localisations_delta, positive_boxes_index), [-1, 4])
      ssd_positive_proposal_targets = tf.reshape(tf.gather(proposal_targets, positive_boxes_index), [-1, 4])
      loss_l1 = smooth_l1(ssd_positive_proposal_predicated - ssd_positive_proposal_targets)
      ssd_loss_box_regression = tf.reduce_mean(tf.reduce_sum(loss_l1, 1), 0, name='regression-loss')
      
      ssd_loss = ssd_cls_cross_entropy + 1.0 * ssd_loss_box_regression
      
      # loss
      tf.losses.add_loss(ssd_loss)
      return ssd_loss
    else:
      rscores, rbboxes = \
        ssd_detected_bboxes(predictions, localisations,
          num_classes=self.num_classes + 1,
          select_threshold=self.select_threshold,
          nms_threshold=self.nms_threshold,
          clipping_bbox=None,
          top_k=self.select_top_k,
          keep_top_k=self.keep_top_k)

      # rscores: N x num x 1
      # rbboxes: N x num x 4
      rscores_list = []
      rbboxes_list = []
      for c in rscores.keys():
        rscores_list.append(tf.expand_dims(rscores[c], axis=1))
        rbboxes_list.append(tf.expand_dims(rbboxes[c], axis=1))
  
      rscores = tf.concat(rscores_list, axis=1)
      rbboxes = tf.concat(rbboxes_list, axis=1)
      return [rscores, rbboxes]
      
##################################################
######## 4.step define training process  #########
##################################################
def unwarp_action(*args, **kwargs):
  # 1.step resize 300 x 300
  image, roidb = args[0]
  height, width = image.shape[0:2]
  standard_img = resize(image, (300, 300))
  
  # 2.step ssd-images, ssd-bbox, ssd-labels
  a = standard_img.reshape((1, 300, 300, 3))
  b = roidb['bbox']
  x_scale = 300.0 / float(width)
  y_scale = 300.0 / float(height)
  b[:,[0,2]] = b[:, [0,2]] * x_scale
  b[:,[1,3]] = b[:, [1,3]] * y_scale
  b = b.reshape((1, -1, 4))
  
  c = roidb['category_id'].reshape((1, -1))
  return a, b, c


def training_callback(data_source, dump_dir):
  ######## 1.step reorganize data ################
  obj_data_source = Node('unwarp', unwarp_action, Node.inputs(data_source))
  
  ######## 2.step building model  ################
  tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=True)
  tf_trainer.deploy(SSDObjDModel())

  ######## 3.step start training  ################
  iter = 0
  for epoch in range(tf_trainer.max_epochs):
    data_generator = obj_data_source.iterator_value()
    while True:
      try:
        _, loss_val = tf_trainer.run(data_generator, {'ssd-images': 0,
                                                      'ssd-bbox': 1,
                                                      'ssd-labels': 2})
        # record loss value
        if iter % 50 == 0:
          loss_channel.send(loss_val)
      
        iter += 1
        
        if iter % 10 == 9:
          break
      except StopIteration:
        break
  
    # save
    tf_trainer.snapshot(epoch)
    
    if epoch == 1:
      break


###################################################
######## 5.step define infer process     ##########
###################################################
def unwarp_test_action(*args, **kwargs):
  # 1.step resize 300 x 300
  image = args[0]
  height, width = image.shape[0:2]
  standard_img = resize(image, (300, 300))
  
  # 2.step ssd-images, ssd-bbox, ssd-labels
  a = standard_img.reshape((1, 300, 300, 3))
  x_scale = 300.0 / float(width)
  y_scale = 300.0 / float(height)

  return a, (x_scale, y_scale)


def infer_callback(data_source, dump_dir):
  ######## 1.step reorganize data ################
  obj_data_source = Node('unwarp', unwarp_test_action, Node.inputs(data_source))
  
  # ######## 2.step building model  ################
  # tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=False)
  # tf_trainer.deploy(SSDObjDModel())

  ######## 3.step start inference  ################
  iter = 0
  data_generator = obj_data_source.iterator_value()
  while True:
    try:
      # res, data = tf_trainer.run(data_generator, {'ssd-images': 0}, whats=True)
      # rscores, rbboxes = res
      # x_scale, y_scale = data[0][1]
      
      next(data_generator)
      
      predict = {}
      bbx1 = np.random.random((10, 1)) * 50
      bby1 = np.random.random((10, 1)) * 50
      bbx2 = bbx1 + np.random.random((10, 1)) * 50
      bby2 = bby1 + np.random.random((10, 1)) * 50
      
      predict['det-bbox'] = np.concatenate((bbx1, bby1, bbx2, bby2), axis=1)
      predict['det-score'] = np.random.random((10, 1))
      predict['det-label'] = np.ones((10,1)) * 2
      
      ctx.recorder.record(predict)
      
      iter += 1
    
      if iter == 20:
        break
    except StopIteration:
      break


###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback