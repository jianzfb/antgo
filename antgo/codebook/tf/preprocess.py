# -*- coding: UTF-8 -*-
# @Time    : 17-12-14
# @File    : preprocess.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim


def tf_regular_augumentation(image,
                             brightness_max_delta=32./255.,
                             saturation_lower=0.5,
                             saturation_upper=1.5,
                             hue_max_delta=0.2,
                             contrast_lower=0.5,
                             contrast_upper=1.5):
  image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
  image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
  image = tf.image.random_hue(image, max_delta=hue_max_delta)
  image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)

  #image = tf.clip_by_value(image, 0, 255)
  return image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  image = control_flow_ops.with_dependencies([size_assertion], tf.slice(image, offsets, cropped_shape))
  return tf.reshape(image, cropped_shape)


def tf_random_crop(image, label, crop_height, crop_width):
  image_rank = tf.rank(image)
  rank_assert = tf.Assert(tf.equal(image_rank, 3),
    ['Wrong rank for tensor  %s [expected] [actual]', image.name, 3, image_rank])
    
  image_shape = control_flow_ops.with_dependencies([rank_assert], tf.shape(image))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.', image_height, image_width, crop_height, crop_width])

  asserts = [rank_assert, crop_size_assert]

  shape = control_flow_ops.with_dependencies([rank_assert], tf.shape(image))
  height = shape[0]
  width = shape[1]

  height_assert = tf.Assert(
      tf.equal(height, image_height),
      ['Wrong height for tensor %s [expected][actual]',
       image.name, height, image_height])
  width_assert = tf.Assert(
      tf.equal(width, image_width),
      ['Wrong width for tensor %s [expected][actual]',
       image.name, width, image_width])
  asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

  cropped_image = _crop(image, offset_height, offset_width, crop_height, crop_width)
  cropped_label = _crop(label, offset_height, offset_width, crop_height, crop_width)
  return cropped_image, cropped_label


def tf_random_rotate(image, label, max_angle=90.0):
  random_angle = (tf.to_float(tf.random_uniform([1]))[0] * 2 - 1) * max_angle / 180.0 * 3.14
  image = tf.expand_dims(image, 0)
  label = tf.expand_dims(label, 0)
  image = tf.contrib.image.rotate(image, random_angle)
  label = tf.contrib.image.rotate(label, random_angle)
  image = tf.squeeze(image, 0)
  label = tf.squeeze(label, 0)
  return image, label


def tf_random_aspect_resize(image, label, low_val=1.0, upper_val=1.5):
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]

  # 1~1.5
  which_side = tf.to_float(tf.random_uniform([1]))[0]
  multi_val = tf.to_float(tf.random_uniform([1]))[0] * (upper_val - low_val) + low_val

  new_height = tf.cond(which_side > 0.5, lambda: tf.to_float(height), lambda: tf.to_float(height) * multi_val)
  new_width = tf.cond(which_side <= 0.5, lambda: tf.to_float(width), lambda: tf.to_float(width) * multi_val)

  new_height = tf.to_int32(new_height)
  new_width = tf.to_int32(new_width)
  
  image = tf.expand_dims(image, 0)
  label = tf.expand_dims(label, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
  resized_image = tf.cast(resized_image, tf.uint8)
  resized_label = tf.image.resize_nearest_neighbor(label, [new_height, new_width], align_corners=False)
  resized_label = tf.cast(resized_label, tf.uint8)
  resized_image = tf.squeeze(resized_image, 0)
  resized_label = tf.squeeze(resized_label, 0)
  return resized_image, resized_label


def _smallest_size_at_least(height, width, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
  
  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)
  
  scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def tf_aspect_preserving_resize(image, label, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
  
  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  
  new_height = tf.maximum(new_height, smallest_side)
  new_width = tf.maximum(new_width, smallest_side)
  
  image = tf.expand_dims(image, 0)
  label = tf.expand_dims(label, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
  resized_image = tf.cast(resized_image, tf.uint8)
  resized_image = tf.squeeze(resized_image, 0)
  
  resized_label = tf.image.resize_nearest_neighbor(label, [new_height, new_width], align_corners=False)
  resized_label = tf.cast(resized_label, tf.uint8)
  resized_label = tf.squeeze(resized_label, 0)
  return resized_image, resized_label


def tf_mirror(image, label):
  val_lr = tf.to_float(tf.random_uniform([1]))[0]
  image = tf.cond(val_lr > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
  label = tf.cond(val_lr > 0.5, lambda: tf.image.flip_left_right(label), lambda: label)
  
  return image, label
