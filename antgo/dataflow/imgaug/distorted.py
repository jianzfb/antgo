# encoding=utf-8
# @Time    : 17-6-8
# @File    : distorted.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.core import *
from antgo.utils.bboxes import *
import copy
import random
import math


class Distorted(Node):
  def __init__(self, inputs,aspect_ratio_range=(0.6, 1.67), area_range=(0.1, 1.0), max_try_times=100):
    super(Distorted, self).__init__(name=None, action=self.action, inputs=inputs)

    self._aspect_ratio_range = aspect_ratio_range
    self._area_range = area_range
    self._max_try_times = max_try_times
  
  def _sample_distorted_bounding_box_like(self, image_size,
                                         bounding_boxes,
                                         aspect_ratio_range=(0.6, 1.67),
                                         area_range=(0.1, 1.0),
                                         max_try_times=100):
    width = image_size[1]
    height = image_size[0]

    box_num = bounding_boxes.shape[0]
    random_box_index = int(np.floor(random.random() * box_num))
    focus_box = bounding_boxes[random_box_index, :]
    focus_box_x0 = focus_box[0] / float(width)
    focus_box_y0 = focus_box[1] / float(height)
    focus_box_x1 = focus_box[2] / float(width)
    focus_box_y1 = focus_box[3] / float(height)
    focus_box_w = focus_box_x1 - focus_box_x0
    focus_box_h = focus_box_y1 - focus_box_y0
    try_time = 0
    w = 1.0
    h = 1.0
    while try_time < max_try_times:
      area = random.random() * (area_range[1] - area_range[0]) + area_range[0]
      aspect_ratio = random.random() * (aspect_ratio_range[1] - aspect_ratio_range[0]) + aspect_ratio_range[0]

      h = math.sqrt((float(width) * area) / (float(height) * aspect_ratio))
      w = area / h

      if w >= focus_box_w and h >= focus_box_h:
          break

      try_time += 1

    if try_time == max_try_times:
      w = 1.0
      h = 1.0

    random_x0_max = focus_box_x0
    random_x0_min = np.maximum(focus_box_x0 - (w - focus_box_w), 0)

    random_y0_max = focus_box_y0
    random_y0_min = np.maximum(focus_box_y0 - (h - focus_box_h), 0)

    x0 = random.random() * (random_x0_max - random_x0_min) + random_x0_min
    y0 = random.random() * (random_y0_max - random_y0_min) + random_y0_min

    begin = np.maximum(np.array((int(x0 * width), int(y0 * height), 0)), np.array((0, 0, 0)))

    w = np.minimum(1.0 - x0, w)
    h = np.minimum(1.0 - y0, h)
    size = np.minimum(np.array((int(w * width), int(h * height), -1)), np.array([width - 1, height - 1, -1]))
    return begin, size, focus_box
  
  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    data, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    assert('bbox' in annotation)

    # distorted bbox
    begin, size, distoted_bbox = self._sample_distorted_bounding_box_like(data.shape,
                                                                          annotation['bbox'],
                                                                          aspect_ratio_range=(0.95, 1.0),
                                                                          area_range=(0.1, 0.3))

    # crop image
    crop_image = data[begin[1]:begin[1]+size[1], begin[0]:begin[0]+size[0]]
    # remained bboxes
    remained_bboxes,remained_bboxes_ind = \
      bboxes_filter_overlap(np.array((begin[0], begin[1], begin[0]+size[0], begin[1]+size[1])),
                            annotation['bbox'], 0.5)

    bboxes = bboxes_translate(np.array((begin[0], begin[1], begin[0]+size[0], begin[1]+size[1])),
                              remained_bboxes)


    # modify annotation
    annotation['bbox'] = bboxes
    if 'category_id' in annotation:
      annotation['category_id'] = annotation['category_id'][remained_bboxes_ind]
    if 'category' in annotation:
      annotation['category'] = [annotation['category'][i] for i in remained_bboxes_ind]
    if 'area' in annotation:
      annotation['area'] = annotation['area'][remained_bboxes_ind]
    if 'info' in annotation:
      annotation['info'] = (crop_image.shape[0], crop_image.shape[1], crop_image.shape[2])

    if 'segmentation' in annotation:
      annotation['segmentation'] = \
          [annotation['segmentation'][i][begin[1]:begin[1]+size[1], begin[0]:begin[0]+size[0]]
           for i in remained_bboxes_ind]
    annotation['cell'] = np.array((begin[0], begin[1], begin[0]+size[0], begin[1]+size[1]))

    return (crop_image, annotation)