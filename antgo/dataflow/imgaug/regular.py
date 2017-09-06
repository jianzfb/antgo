# encoding=utf-8
# @Time    : 17-6-7
# @File    : regular.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import scipy.misc
from antgo.utils import get_rng
from antgo.dataflow.core import *
import copy
from scipy.ndimage.interpolation import affine_transform


class Flip(Node):
  def __init__(self, inputs, horiz=False, vert=False, prob=0.5):
    super(Flip, self).__init__(name=None, action=self.action, inputs=inputs)

    """
    Only one of horiz, vert can be set.

    :param horiz: whether or not apply horizontal flip.
    :param vert: whether or not apply vertical flip.
    :param prob: probability of flip.
    """
    if horiz and vert:
      raise ValueError("Please use two Flip instead.")
    elif horiz:
      self.code = 1
    elif vert:
      self.code = 0
    else:
      raise ValueError("Are you kidding?")
    self.prob = prob
    self.rng = get_rng(self)

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    is_flip = self.rng.uniform(0, 1) < self.prob
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    annotation = copy.deepcopy(annotation)
    if is_flip:
      # for image
      if self.code == 1:
        image = np.fliplr(image)
      else:
        image = np.flipud(image)

      # for annotation
      if 'bbox' in annotation:
        if self.code == 1:
          w = annotation['info'][1]
          boxes = annotation['bbox'].copy()
          oldx1 = boxes[:, 0].copy()
          oldx2 = boxes[:, 2].copy()
          boxes[:, 0] = w - oldx2 - 1
          boxes[:, 2] = w - oldx1 - 1

          assert ((boxes[:, 2] >= boxes[:, 0]).all())
          annotation['bbox'] = boxes
          annotation['flipped'] = True
        else:
          h = annotation['info'][0]
          boxes = annotation['bbox'].copy()
          oldy1 = boxes[:, 1].copy()
          oldy2 = boxes[:, 3].copy()
          boxes[:, 1] = h - oldy2 - 1
          boxes[:, 3] = h - oldy1 - 1

          assert ((boxes[:, 3] >= boxes[:, 1]).all())
          annotation['bbox'] = boxes
          annotation['flipped'] = True
      if 'segmentation' in annotation:
        resized_obj_seg = []
        for obj_seg in annotation['segmentation']:
          temp = None
          if self.code == 1:
            temp = np.fliplr(obj_seg)
          else:
            temp = np.flipud(obj_seg)
          resized_obj_seg.append(temp)
        annotation['segmentation'] = resized_obj_seg

    return (image, annotation)


class Resize(Node):
  def __init__(self, inputs, shape):
    super(Resize, self).__init__(name=None, action=self.action, inputs=inputs)
    self.shape = shape

  def action(self, *args, **kwargs):
    assert (len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    # for image
    image = scipy.misc.imresize(image, self.shape)
    annotation = copy.deepcopy(annotation)
    # for annotation
    if type(annotation) == dict and 'bbox' in annotation:
      boxes = annotation['bbox']
      info = annotation['info']
      horizontal_scale = float(self.shape[1]) / float(info[1])
      vertical_scale = float(self.shape[0]) / float(info[0])

      boxes[:, [0, 2]] = boxes[:, [0, 2]] * horizontal_scale
      boxes[:, [1, 3]] = boxes[:, [1, 3]] * vertical_scale

      annotation['bbox'] = boxes
      annotation['info'] = (self.shape[0], self.shape[1], info[2])
    if type(annotation) == dict and 'segmentation' in annotation:
      resized_obj_seg = []
      for obj_seg in annotation['segmentation']:
        temp = scipy.misc.imresize(obj_seg[:, :, 0], self.shape[::-1], 'nearest')
        resized_obj_seg.append(temp.reshape(temp.shape[0],temp.shape[1],1))
      annotation['segmentation'] = resized_obj_seg

    return (image, annotation)


class Subtract(Node):
  def __init__(self, inputs, mean_val=(128, 128, 128)):
    super(Subtract, self).__init__(name=None, action=self.action, inputs=inputs)
    self._mean_value = mean_val

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})

    # for image
    image = image - np.array(self._mean_value)

    return (image, annotation)


class DisturbChannels(Node):
  def __init__(self, inputs):
    super(DisturbChannels, self).__init__(name=None, action=self.action, inputs=inputs)

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    if len(image.shape) < 3:
      return [image, annotation]

    channles = np.arange(0, image.shape[2])
    np.random.shuffle(channles)
    image_pieces = np.split(image, len(channles), axis=2)
    image = np.concatenate([image_pieces[i] for i in channles], axis=2)

    return (image, annotation)


class DisturbRotation(Node):
  def __init__(self, inputs, max_deg=10):
    super(DisturbRotation, self).__init__(name=None, action=self.action, inputs=inputs)
    self._max_deg = max_deg
    self.rng = get_rng(self)

  @staticmethod
  def _affine_matrix(rotate_center, rotate_deg):
    theta = np.deg2rad(rotate_deg)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # offset = np.array(rotate_center) - np.array(rotate_center).dot(rot)
    offset = (rotate_center - rotate_center.dot(rot)).dot(np.linalg.inv(rot))

    return rot, -offset

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})

    deg = self.rng.uniform(-self._max_deg, self._max_deg)
    center_x = image.shape[1] / 2.0
    center_y = image.shape[0] / 2.0
    rot, offset = DisturbRotation._affine_matrix(np.array([center_y, center_x]), -deg)

    image_channles = [None, None, None]
    for channel, image_channle in enumerate(np.split(image, 3, axis=2)):
      image_channles[channel] = \
          affine_transform(np.squeeze(image_channle, 2),
                           rot,
                           order=3,
                           offset=offset,
                           cval=0,
                           output=np.uint8)

    image = np.concatenate((np.expand_dims(image_channles[0], 2),
                            np.expand_dims(image_channles[1], 2),
                            np.expand_dims(image_channles[2], 2)), axis=2)
    
    annotation_cpy = copy.deepcopy(annotation)
    if type(annotation_cpy) == dict and 'bbox' in annotation_cpy:
      bbox = annotation_cpy['bbox']

      rrr_theta = np.deg2rad(deg)
      rrr = np.array([[np.cos(rrr_theta), np.sin(rrr_theta)], [-np.sin(rrr_theta), np.cos(rrr_theta)]])

      # x0,y0,x1,y1,x0,y1,x1,y0
      bbox_mm = np.concatenate((bbox,
                                bbox[:, 0][:, np.newaxis], bbox[:, 3][:, np.newaxis],
                                bbox[:, 2][:, np.newaxis], bbox[:, 1][:, np.newaxis]), axis=1)
      bbox_mm = bbox_mm.reshape(-1, 2)
      rotate_bbox = np.matmul(bbox_mm - np.array([center_x, center_y]).reshape(-1, 2), rrr) + \
                    np.array([center_x, center_y]).reshape(-1, 2)
      rotate_bbox = rotate_bbox.reshape(-1, 8)

      # reset bbox and clip
      temp = np.zeros(bbox.shape)
      for bbox_i, bbox in enumerate(rotate_bbox):
        x0, y0, x1, y1, xx0, yy0, xx1, yy1 = bbox
        n_x0 = np.minimum(np.maximum(np.min([x0, x1, xx0, xx1]), 0), image.shape[1] - 1)
        n_y0 = np.minimum(np.maximum(np.min([y0, y1, yy0, yy1]), 0), image.shape[0] - 1)

        n_x1 = np.maximum(np.minimum(np.max([x0, x1, xx0, xx1]), image.shape[1] - 1), 0)
        n_y1 = np.maximum(np.minimum(np.max([y0, y1, yy0, yy1]), image.shape[0] - 1), 0)

        temp[bbox_i, 0] = n_x0
        temp[bbox_i, 1] = n_y0
        temp[bbox_i, 2] = n_x1
        temp[bbox_i, 3] = n_y1

      annotation_cpy['bbox'] = temp

    if type(annotation_cpy) == dict and 'segmentation' in annotation_cpy:
      obj_segs = []
      for obj_seg in annotation_cpy['segmentation']:
        obj_segs.append(
          np.expand_dims(affine_transform(np.squeeze(obj_seg, 2),
                                          rot,
                                          order=3,
                                          offset=offset,
                                          cval=0,
                                          output=np.uint8), axis=2))

      annotation_cpy['segmentation'] = obj_segs
    return image, annotation_cpy


class DisturbLighting(Node):
  def __init__(self, inputs, max_lighting=10):
    super(DisturbLighting, self).__init__(name=None, action=self.action, inputs=inputs)
    self._max_lighting = max_lighting
    self.rng = get_rng(self)

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    light = self.rng.uniform(-self._max_lighting, self._max_lighting)
    image_cpy = image.copy() + light

    return image_cpy, annotation


class DisturbNoise(Node):
  def __init__(self, inputs, sigma=10):
    super(DisturbNoise, self).__init__(name=None, action=self.action, inputs=inputs)
    self._sigma = sigma
    self.rng = get_rng(self)

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    image, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    noise = (np.random.random((height, width, channels)) * 2 - 1) * self._sigma
    image_cpy = image.copy() + noise

    return image_cpy, annotation
