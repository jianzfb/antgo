# encoding=utf-8
# @Time    : 17-5-16
# @File    : ssd.py
# @Author  : Z<zhangjian8@xiaomi.com>
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
import tf_extended as tfe
from nets import ssd_common
from nets import custom_layers
from antgo.utils.helper import *
from collections import namedtuple
import math
def tf_bbox_transform_inv(boxes, deltas, name):
  return tf.py_func(bbox_transform_inv, [boxes, deltas], [tf.float32], name=name)


def tf_bbox_clip(boxes, im_shape, name):
  return tf.py_func(clip_boxes, [boxes, im_shape], [tf.float32], name=name)


def tensor_shape(x, rank=3):
  """Returns the dimensions of a tensor.
  Args:
    image: A N-D Tensor of shape.
  Returns:
    A list of dimensions. Dimensions that are statically known are python
      integers,otherwise they are integer scalar tensors.
  """
  if x.get_shape().is_fully_defined():
    return x.get_shape().as_list()
  else:
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
  """Construct a multibox layer, return a class and localization predictions.
  """
  net = inputs
  if normalization > 0:
    net = custom_layers.l2_normalization(net, scaling=True)
  # Number of anchors.
  num_anchors = len(sizes) + len(ratios)

  # Location.
  num_loc_pred = num_anchors * 4
  loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                         scope='conv_loc')
  loc_pred = tf.reshape(loc_pred,
                        tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
  # Class prediction.
  num_cls_pred = num_anchors * num_classes
  cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                         scope='conv_cls')
  cls_pred = tf.reshape(cls_pred,
                        tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
  return cls_pred, loc_pred


SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

"""Implementation of the SSD VGG-based 300 network.

The default features layers with 300x300 image input are:
  conv4 ==> 38 x 38
  conv7 ==> 19 x 19
  conv8 ==> 10 x 10
  conv9 ==> 5 x 5
  conv10 ==> 3 x 3
  conv11 ==> 1 x 1
The default image size used to train this network is 300x300.
"""
default_params = SSDParams(
    img_shape=(300, 300),
    num_classes=21,
    no_annotation_label=21,
    feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_size_bounds=[0.20, 0.90],
    anchor_sizes=[(21., 45.),
                  (45., 99.),
                  (99., 153.),
                  (153., 207.),
                  (207., 261.),
                  (261., 315.)],
    anchor_ratios=[[2, .5],
                   [2, .5, 3, 1. / 3],
                   [2, .5, 3, 1. / 3],
                   [2, .5, 3, 1. / 3],
                   [2, .5],
                   [2, .5]],
    anchor_steps=[8, 16, 32, 64, 100, 300],
    anchor_offset=0.5,
    normalizations=[20, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2]
)


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        padding='SAME',
                        data_format=data_format):
      with slim.arg_scope([custom_layers.pad2d,
                           custom_layers.l2_normalization,
                           custom_layers.channel_to_last],
                          data_format=data_format) as sc:
        return sc


def ssd(data, num_classes, is_training, reuse=None, scope='ssd_300_vgg'):
  end_points = {}
  with tf.variable_scope(scope, 'ssd_300_vgg', [data], reuse=reuse):
    # Original VGG-16 blocks.
    net = slim.repeat(data, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    end_points['block1'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    # Block 2.
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    end_points['block2'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    # Block 3.
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    end_points['block3'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    # Block 4.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    end_points['block4'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    # Block 5.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    end_points['block5'] = net
    net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

    # Additional SSD blocks.
    # Block 6: let's dilate the hell out of it!
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
    end_points['block6'] = net
    net = tf.layers.dropout(net, rate=0.5, training=is_training)
    # Block 7: 1x1 conv. Because the fuck.
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    end_points['block7'] = net
    net = tf.layers.dropout(net, rate=0.5, training=is_training)

    # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
    end_point = 'block8'
    with tf.variable_scope(end_point):
      net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
      net = custom_layers.pad2d(net, pad=(1, 1))
      net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block9'
    with tf.variable_scope(end_point):
      net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
      net = custom_layers.pad2d(net, pad=(1, 1))
      net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block10'
    with tf.variable_scope(end_point):
      net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block11'
    with tf.variable_scope(end_point):
      net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points[end_point] = net

    # Prediction and localisations layers.
    predictions = []
    logits = []
    localisations = []
    for i, layer in enumerate(default_params.feat_layers):
      with tf.variable_scope(layer + '_box'):
        p, l = ssd_multibox_layer(end_points[layer],
                                  num_classes,
                                  default_params.anchor_sizes[i],
                                  default_params.anchor_ratios[i],
                                  default_params.normalizations[i])
      predictions.append(slim.softmax(p))
      logits.append(p)
      localisations.append(l)

  return predictions, logits, localisations


def ssd_bbox_decoder(default_anchors, localisations):
  # 4.step make proposals
  index = 0
  proposals = []
  for a, b in zip(default_anchors, localisations):
    y, x, h, w = a
    shifts = np.vstack((x.ravel(), y.ravel(),
                        x.ravel(), y.ravel())).transpose()

    A = h.shape[0]
    windows = np.vstack((np.zeros((A)), np.zeros(A), w, h)).transpose()
    K = shifts.shape[0]
    shift_anchors = windows.reshape((1, A, 4)) + \
                    shifts.reshape((1, K, 4)).transpose((1, 0, 2))

    shift_anchors = shift_anchors.reshape(-1,4)
    shift_anchors = shift_anchors * 300.0
    layer_proposals = tf_bbox_transform_inv(shift_anchors, tf.reshape(b,(-1,4)), '%s_proposal'%default_params.feat_layers[index])
    layer_clip_proposals = tf_bbox_clip(layer_proposals, (300, 300), '%s_proposal_clip'%default_params.feat_layers[index])
    index += 1
    layer_clip_proposals = tf.reshape(layer_clip_proposals, (-1, K * A, 4))
    proposals.append(layer_clip_proposals)

  return proposals


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
  """Computer SSD default anchor boxes for one feature layer.

  Determine the relative position grid of the centers, and the relative
  width and height.

  Arguments:
    feat_shape: Feature shape, used for computing relative position grids;
    size: Absolute reference sizes;
    ratios: Ratios to use on these features;
    img_shape: Image shape, used for computing height, width relatively to the
      former;
    offset: Grid offset.

  Return:
    y, x, h, w: Relative x and y grids, and height and width.
  """
  # Compute the position grid: simple way.
  # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
  # y = (y.astype(dtype) + offset) / feat_shape[0]
  # x = (x.astype(dtype) + offset) / feat_shape[1]
  # Weird SSD-Caffe computation using steps values...
  y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
  y = (y.astype(dtype) + offset) * step / img_shape[0]
  x = (x.astype(dtype) + offset) * step / img_shape[1]

  # Expand dims to support easy broadcasting.
  y = np.expand_dims(y, axis=-1)
  x = np.expand_dims(x, axis=-1)

  # Compute relative height and width.
  # Tries to follow the original implementation of SSD for the order.
  num_anchors = len(sizes) + len(ratios)
  h = np.zeros((num_anchors, ), dtype=dtype)
  w = np.zeros((num_anchors, ), dtype=dtype)
  # Add first anchor boxes with ratio=1.
  h[0] = sizes[0] / img_shape[0]
  w[0] = sizes[0] / img_shape[1]
  di = 1
  if len(sizes) > 1:
    h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
    w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
    di += 1
  for i, r in enumerate(ratios):
    h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
    w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
  return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
  """Compute anchor boxes for all feature layers.
  """
  layers_anchors = []
  for i, s in enumerate(layers_shape):
    anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                         anchor_sizes[i],
                                         anchor_ratios[i],
                                         anchor_steps[i],
                                         offset=offset, dtype=dtype)
    layers_anchors.append(anchor_bboxes)
  return layers_anchors


def ssd_anchors(img_shape, dtype=np.float32):
  """Compute the default anchor boxes, given an image shape.
  """
  return ssd_anchors_all_layers(img_shape,
                                default_params.feat_shapes,
                                default_params.anchor_sizes,
                                default_params.anchor_ratios,
                                default_params.anchor_steps,
                                default_params.anchor_offset,
                                dtype)


def ssd_detected_bboxes(predictions, localisations, num_classes,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
  """Get the detected bounding boxes from the SSD network output.
  """
  # Select top_k bboxes from predictions, and clip
  rscores, rbboxes = \
    ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                    select_threshold=select_threshold,
                                    num_classes=num_classes)
  rscores, rbboxes = \
    tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
  # Apply NMS algorithm.
  rscores, rbboxes = \
    tfe.bboxes_nms_batch(rscores, rbboxes,
                         nms_threshold=nms_threshold,
                         keep_top_k=keep_top_k)
  if clipping_bbox is not None:
    rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
  return rscores, rbboxes
