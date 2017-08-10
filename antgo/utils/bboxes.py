# encoding=utf-8
# @Time    : 17-5-26
# @File    : bboxes.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np


def bboxes_translate(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """

    # translate
    v = np.stack([bbox_ref[0],bbox_ref[1],bbox_ref[0],bbox_ref[1]])
    bboxes = bboxes - v
    bboxes[:,0] = np.maximum(bboxes[:,0],0)
    bboxes[:,1] = np.maximum(bboxes[:,1],0)
    bboxes[:,2] = np.minimum(bboxes[:,2],bbox_ref[2] - bbox_ref[0])
    bboxes[:,3] = np.minimum(bboxes[:,3],bbox_ref[3] - bbox_ref[1])
    return bboxes


def bboxes_filter_overlap(bbox_ref,bboxes,min_overlap = 0.5):
    bboxes_num = bboxes.shape[0]
    bbox_ref_tile = np.tile(bbox_ref,[bboxes_num,1])

    union_bboxes_x0 = np.maximum(bbox_ref_tile[:, 0], bboxes[:, 0])
    union_bboxes_y0 = np.maximum(bbox_ref_tile[:, 1], bboxes[:, 1])
    union_bboxes_x1 = np.minimum(bbox_ref_tile[:, 2], bboxes[:, 2])
    union_bboxes_y1 = np.minimum(bbox_ref_tile[:, 3], bboxes[:, 3])
    union_bboxes_area = (union_bboxes_x1 - union_bboxes_x0) * (union_bboxes_y1 - union_bboxes_y0)
    normalized_boxes_area = (bboxes[:, 2] - bboxes[:, 0]) * \
                            (bboxes[:, 3] - bboxes[:, 1])

    overlaps = union_bboxes_area / normalized_boxes_area
    invalid_index_1 = np.where(union_bboxes_x1 - union_bboxes_x0 < 0)[0]
    invalid_index_2 = np.where(union_bboxes_y1 - union_bboxes_y0 < 0)[0]
    invalid_index = list(set(invalid_index_1) & set(invalid_index_2))
    overlaps[invalid_index] = -1.0
    valid_bboxes_index = np.where(overlaps > min_overlap)[0]
    return bboxes[valid_bboxes_index,:],valid_bboxes_index
