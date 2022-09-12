# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import logging
import cv2
import numpy as np
from antgo.dataflow.core import *
from .operators import BaseOperator
from .op_helper import jaccard_overlap, gaussian2D



logger = logging.getLogger(__name__)


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True, inputs=None):
        super(PadBatch, self).__init__(inputs=inputs)
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem

        return samples


class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False, inputs=None):
        super(RandomShape, self).__init__(inputs=inputs)
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
            if self.resize_box and 'gt_bbox' in samples[i] and len(samples[0][
                    'gt_bbox']) > 0:
                scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
                samples[i]['gt_bbox'] = np.clip(samples[i]['gt_bbox'] *
                                                scale_array, 0,
                                                float(shape) - 1)
        return samples


class PadMultiScaleTest(BaseOperator):
    """
    Pad the image so they can be divisible by a stride for multi-scale testing.
 
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, inputs=None):
        super(PadMultiScaleTest, self).__init__(inputs=inputs)
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples, context=None):
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples

        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        if len(samples) != 1:
            raise ValueError("Batch size must be 1 when using multiscale test, "
                             "but now batch size is {}".format(len(samples)))
        for i in range(len(samples)):
            sample = samples[i]
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im_c, im_h, im_w = im.shape
                    max_h = int(
                        np.ceil(im_h / coarsest_stride) * coarsest_stride)
                    max_w = int(
                        np.ceil(im_w / coarsest_stride) * coarsest_stride)
                    padding_im = np.zeros(
                        (im_c, max_h, max_w), dtype=np.float32)

                    padding_im[:, :im_h, :im_w] = im
                    sample[k] = padding_im
                    info_name = 'im_info' if k == 'image' else 'im_info_' + k
                    # update im_info
                    sample[info_name][:2] = [max_h, max_w]
        if not batch_input:
            samples = samples[0]
        return samples


class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operators is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.,
                 inputs=None):
        super(Gt2YoloTarget, self).__init__(inputs=inputs)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match 
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou 
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target
        return samples


class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False,
                 inputs=None):
        super(Gt2FCOSTarget, self).__init__(inputs=inputs)
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2
        beg = 0
        clipped_box = bboxes.copy()
        for lvl, stride in enumerate(self.downsample_ratios):
            end = beg + num_points_each_level[lvl]
            stride_exp = self.center_sampling_radius * stride
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)
            beg = end
        l_res = xs - clipped_box[:, :, 0]
        r_res = clipped_box[:, :, 2] - xs
        t_res = ys - clipped_box[:, :, 1]
        b_res = clipped_box[:, :, 3] - ys
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                np.floor(im_info[1] / im_info[2])
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]
            points, num_points_each_level = self._compute_points(w, h)
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):
                object_scale_exp.append(
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (
                bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])

            l_res = xs - bboxes[:, 0]
            r_res = bboxes[:, 2] - xs
            t_res = ys - bboxes[:, 1]
            b_res = bboxes[:, 3] - ys
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
            if self.center_sampling_radius > 0:
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)
            lower_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF
            points2gtarea[is_match_current_level == 0] = self.INF
            points2min_area = points2gtarea.min(axis=1)
            points2min_area_ind = points2gtarea.argmin(axis=1)
            labels = gt_class[points2min_area_ind] + 1
            labels[points2min_area == self.INF] = 0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                  reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])
            ctn_targets[labels <= 0] = 0
            pos_ind = np.nonzero(labels != 0)
            reg_targets_pos = reg_targets[pos_ind[0], :]
            split_sections = []
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            labels_by_level = np.split(labels, split_sections, axis=0)
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))
                if self.norm_reg_targets:
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],
                            newshape=[grid_h, grid_w, 4])
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])
        return samples


class Gt2TTFTarget(BaseOperator):
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, num_classes, down_ratio=4, alpha=0.54, inputs=None):
        super(Gt2TTFTarget, self).__init__(inputs=inputs)
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, samples, context=None):
        output_size = samples[0]['image'].shape[1]
        feat_size = output_size // self.down_ratio
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size, feat_size), dtype='float32')
            box_target = np.ones(
                (4, feat_size, feat_size), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size, feat_size), dtype='float32')

            '''
            if "gt_keypoint" in sample:
                continue
            '''
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']

            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            boxes_area_topk_log = boxes_areas_log[boxes_ind]
            gt_bbox = gt_bbox[boxes_ind]
            gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)
            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype('int32')

            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                fake_heatmap = np.zeros((feat_size, feat_size), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k],
                                            h_radiuses_alpha[k],
                                            w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div
            sample['ttf_heatmap'] = heatmap
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius

    def __call__(self, joints, area):
        assert joints.shape[1] == self.num_joints_with_center, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)

        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue

            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                                
                            offset_map[idx*2, pos_y, pos_x] = offset_x          # normalize
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / (np.sqrt(area[person_id]) + 1e-6)
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / (np.sqrt(area[person_id]) + 1e-6)
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map

        

class PafWithCenterGenerator():
    def __init__(self, output_h, output_w, num_joints, width=1.0, is_reduce=True):
        self.output_h = output_h
        self.output_w = output_w
        self.width = width
        self.num_joints = num_joints

    def __call__(self, joints, area):
        # ignore area
        feat_size_h = self.output_h
        feat_size_w = self.output_w
        heatmap_paf = np.zeros((feat_size_h, feat_size_w, 2), dtype='float32')
        count = np.zeros((int(feat_size_h), int(feat_size_w)), dtype=np.uint32)

        for person_id, p in enumerate(joints):
            ct_x, ct_y, ct_v = p[ -1, :3]
            if (ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h):
                continue

            for idx, pt in enumerate(p[:-1]):
                heatmap_paf, count = putVecMaps(
                    centerA=p[-1,:2],
                    centerB=pt[:2],
                    accumulate_vec_map=heatmap_paf,
                    count=count,
                    grid_y=feat_size_h, grid_x=feat_size_w,
                    stride=1,
                    width=self.width
                )

        heatmap_paf = np.transpose(heatmap_paf, (2,0,1))
        return heatmap_paf


class Gt2TTFKPTTarget(Gt2TTFTarget):
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    
    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, 
                num_classes, 
                down_ratio=4, 
                alpha=0.54, 
                num_kpts=21, 
                has_offset=False,
                has_paf=False, 
                kpt_down_scale=2, 
                inputs=None):
        super(Gt2TTFKPTTarget, self).__init__(
                num_classes=num_classes,
                down_ratio=down_ratio,
                alpha=alpha,
                inputs=inputs)
        self.num_kpts = num_kpts
        self.offset_generator = None
        if has_offset:
            self.offset_generator = OffsetGenerator(160//4, 192//4, 21, 4)
        
        self.paf_generator = None
        if has_paf:
            self.paf_generator = \
                PafWithCenterGenerator(output_h=160//4, 
                                       output_w=192//4,
                                       num_joints=21, 
                                       width=1.0)
        
        self.kpt_down_scale = kpt_down_scale
        
    def __call__(self, samples, context=None):
        # output_size = samples[0]['image'].shape[1]
        output_size_h, output_size_w = samples[0]['image'].shape[1:3]
        feat_size_h = output_size_h // self.down_ratio
        feat_size_w = output_size_w // self.down_ratio
        heatmap_kpt_h = output_size_h // self.kpt_down_scale
        heatmap_kpt_w = output_size_w // self.kpt_down_scale
        
        count = 0
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size_h, feat_size_w), dtype='float32')
            heatmap_kpt = np.zeros(
                (self.num_kpts, heatmap_kpt_h, heatmap_kpt_w), dtype='float32')            
            heatmap_one = np.zeros(
                (1, feat_size_h, feat_size_w), dtype='float32')
            box_target = np.ones(
                (4, feat_size_h, feat_size_w), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size_h, feat_size_w), dtype='float32')     

            # heatmap mask valid
            heatmap_mask = np.ones((1, feat_size_h, feat_size_w), dtype='float32')

            # henggang kpt
            henggang_kpt = np.zeros((4, heatmap_kpt_h, heatmap_kpt_w), dtype='float32')

            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            # boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            # boxes_area_topk_log = boxes_areas_log[boxes_ind]
            # gt_bbox = gt_bbox[boxes_ind]
            # gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            # feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)

            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio
            
            h_radiuses_alpha = (feat_hs / 2. * self.alpha + 0.5).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha + 0.5).astype('int32')
            # boxes_scale = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1]) * \
            #                   (feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0]) / \
            #                   ((feat_size_w - 1) * (feat_size_h - 1))

            # num_obj = ct_inds.shape[0]
            # ct_inds_ext = np.concatenate([ct_inds, np.ones((num_obj, 1))], axis=1)
            # joints = np.zeros((num_ins, self.num_kpts, 3))
            # joints = np.concatenate([joints, np.expand_dims(ct_inds_ext, 1)], axis=1)       # num_ins, self.num_kpts+1,3
            # person_areas = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1]) * (feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])
            
            body_bbox = []
            person_num = 0
            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                if cls_id == 0:             # 人体类别 0
                    person_num += 1
                    body_bbox.append(gt_bbox[k].copy())
                
                fake_heatmap = np.zeros((feat_size_h, feat_size_w), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k], h_radiuses_alpha[k], w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_areas_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div

            joints = np.zeros((person_num, self.num_kpts+1, 3))
            person_areas = np.zeros((person_num))
            if 'gt_keypoint' in sample:
                gt_kpts = sample['gt_keypoint'].copy()
                gt_kpts_for_heatmap = sample['gt_keypoint'].copy()
                gt_kpts[:,:,:2] = gt_kpts[:,:,:2] / self.down_ratio
                gt_kpts_for_heatmap[:,:,:2] = gt_kpts_for_heatmap[:,:,:2] / self.kpt_down_scale
                gt_kpts_from_bbox_i = sample['gt_keypoint_from_bbox_i']
                
                num_ins = gt_kpts.shape[0]                
                joints[:, :-1, :] = gt_kpts
                
                # 计算每个人的中心
                for k in range(num_ins):
                    from_bbox_i = gt_kpts_from_bbox_i[k]
                    joints[k, -1, 0] = (feat_gt_bbox[from_bbox_i, 0] + feat_gt_bbox[from_bbox_i, 2])/2.0
                    joints[k, -1, 1] = (feat_gt_bbox[from_bbox_i, 1] + feat_gt_bbox[from_bbox_i, 3])/2.0
                    joints[k, -1, 2] = 1.0                    
                
                # 计算每个人的面积
                for k in range(num_ins):
                    person_min_x = 10000
                    person_max_x = 0
                    person_min_y = 10000
                    person_max_y = 0
                    for jter in range(self.num_kpts):
                        if gt_kpts[k, jter, 2] < 0.001:
                            # 忽略无效关键点
                            continue
                        
                        person_min_x = min(gt_kpts[k, jter, 0], person_min_x)
                        person_max_x = max(gt_kpts[k, jter, 0], person_max_x)
                        
                        person_min_y = min(gt_kpts[k, jter, 1], person_min_y)
                        person_max_y = max(gt_kpts[k, jter, 1], person_max_y)
                    
                    person_areas[k] = max(0, (person_max_x-person_min_x)*(person_max_y-person_min_y))
                        
                # # 构建KPT heatmap
                # # DEBUG:
                # ssss = sample['image'].copy()
                # ssss = np.transpose(ssss, (1,2,0)).astype(np.uint8).copy()
                for k in range(num_ins):
                    for jter in range(self.num_kpts):
                        if gt_kpts_for_heatmap[k, jter, 2] < 0.001:
                            # 忽略无效关键点
                            continue
                        
                        fake_heatmap = np.zeros((heatmap_kpt_h, heatmap_kpt_w), dtype='float32')
                        kpt_pt = gt_kpts_for_heatmap[k, jter, 0:2]
                        # 这里kpt采用固定宽度构建heatmap
                        self.draw_truncate_gaussian(fake_heatmap, kpt_pt, 7, 7)
                        heatmap_kpt[jter] = np.maximum(heatmap_kpt[jter], fake_heatmap)
                        
                        # # DEBUG: 
                        # xx = min((int)(gt_kpts_for_heatmap[k, jter, 0]) * self.kpt_down_scale, 191)
                        # yy = min((int)(gt_kpts_for_heatmap[k, jter, 1]) * self.kpt_down_scale, 159)
                        # cv2.circle(ssss, (xx,yy), 5, (0,0,255))
                        # cv2.putText(ssss, 
                        #                 f"{jter}", 
                        #                 (xx-1, yy-5), 
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 
                        #                 0.3,
                        #                 (0, 255, 0),
                        #                 1)

                heatmap_kpt = np.clip(heatmap_kpt, a_min=0.0, a_max=1.0)
                
                # # # DEBUG: 显示heatmap kpt
                # # cv2.imwrite(f"./temp/s_id_{count}.png", (np.max(heatmap_kpt, 0)*255).astype(np.uint8))
                # cv2.imwrite(f"./temp/s_image_id_{count}.png", ssss)

            sample['ttf_heatmap'] = heatmap
            sample['ttf_kpt_heatmap'] = heatmap_kpt
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
            sample['ttf_ext_kpt_heatmap'] = henggang_kpt
            sample['ttf_heatmap_mask'] = heatmap_mask

            # cv2.imwrite(f"./temp/{count}_left.png", (henggang_kpt[0]*255).astype(np.uint8))
            # cv2.imwrite(f"./temp/{count}_right.png", (henggang_kpt[1]*255).astype(np.uint8))
            body_bbox = np.stack(body_bbox)
            body_bbox = body_bbox.astype(np.float32)
            half_body_bbox_w = (body_bbox[:, 2] - body_bbox[:, 0] + 1)/2.0
            half_body_bbox_h = (body_bbox[:, 3] - body_bbox[:, 1] + 1)/2.0            

            # 引入随机抖动half_size
            half_body_bbox_h = half_body_bbox_h * (np.random.random() * 0.5 + 0.8)
            half_body_bbox_w =  half_body_bbox_w * (np.random.random() * 0.5 + 0.8)
            body_bbox_x = (body_bbox[:, 0] + body_bbox[:, 2]) / 2
            body_bbox_y = (body_bbox[:, 1] + body_bbox[:, 3]) / 2
            # 引入随机抖动cx
            body_bbox_x += (2*np.random.random() - 1.0) * 0.3 * np.minimum(half_body_bbox_w,half_body_bbox_h)
            body_bbox_y += (2*np.random.random() - 1.0) * 0.3 * np.minimum(half_body_bbox_w,half_body_bbox_h)

            body_bbox = [body_bbox_x-half_body_bbox_w, 
                            body_bbox_y-half_body_bbox_h,
                            body_bbox_x+half_body_bbox_w,
                            body_bbox_y+half_body_bbox_h]
            body_bbox = np.stack(body_bbox, 1)

            # 确保固定包含3个框
            body_bbox_num = body_bbox.shape[0]
            body_bbox_select_list = list(range(body_bbox_num))
            np.random.shuffle(body_bbox_select_list)
            if len(body_bbox_select_list) < 3:
                add_list = list(np.random.choice(body_bbox_select_list, 3-len(body_bbox_select_list)))
                body_bbox_select_list = body_bbox_select_list + add_list
            body_bbox_select_list = body_bbox_select_list[:3]

            sample['ttf_roi_box'] = body_bbox[body_bbox_select_list].astype(np.float32)
            sample['ttf_roi_num'] = np.array([len(body_bbox_select_list)])

            sample['ttf_roi_kpt'] = np.zeros((len(body_bbox_select_list), 21, 40, 40), dtype=np.float32)    # Human x 21 x H x W
            gt_kpts_for_heatmap = sample['gt_keypoint'].copy()
            gt_kpts_for_heatmap[:,:,:2] = gt_kpts_for_heatmap[:,:,:2] / self.kpt_down_scale
            gt_kpts_from_bbox_i = sample['gt_keypoint_from_bbox_i']
            
            ttf_roi_kpt_offset = np.zeros((len(body_bbox_select_list), 21, 40, 40), dtype=np.float32)    # Human x 21 x H x W
            # 
            gt_kpts_reg_weight = np.zeros((len(body_bbox_select_list),1, 40, 40), dtype='float32') 
            gt_kpts_reg = np.ones((len(body_bbox_select_list), 2, 40, 40), dtype='float32') * -1

            for index, human_id in enumerate(body_bbox_select_list):
                human_x0,human_y0,human_x1,human_y1 = body_bbox[human_id] / self.kpt_down_scale
                human_w = human_x1 - human_x0
                human_h = human_y1 - human_y0
                
                for jter in range(self.num_kpts):
                    if gt_kpts_for_heatmap[human_id, jter, 2] < 0.001 or \
                        (gt_kpts_for_heatmap[human_id, jter, 0] < human_x0 or  gt_kpts_for_heatmap[human_id, jter, 1] < human_y0) or \
                        (gt_kpts_for_heatmap[human_id, jter, 0] > human_x1 or  gt_kpts_for_heatmap[human_id, jter, 1] > human_y1):
                        # 忽略无效关键点
                        continue
                    
                    kpt_pt = gt_kpts_for_heatmap[human_id, jter, 0:2]
                    joint_kpt_x, joint_kpt_y = kpt_pt
                    joint_kpt_x = (joint_kpt_x - human_x0) / human_w * 40.0
                    joint_kpt_y = (joint_kpt_y - human_y0) / human_h * 40.0
                    kpt_pt = np.array([joint_kpt_x, joint_kpt_y])

                    # for heatmap
                    fake_heatmap = np.zeros((40, 40), dtype='float32')                    
                    self.draw_truncate_gaussian(fake_heatmap, kpt_pt, 3, 3)
                    sample['ttf_roi_kpt'][index, jter] = np.maximum(sample['ttf_roi_kpt'][index, jter], fake_heatmap)

                    # for offset
                    fake_heatmap = np.zeros((40, 40), dtype='float32')
                    self.draw_truncate_gaussian(fake_heatmap, kpt_pt, 5, 5)
                    ttf_roi_kpt_offset[index, jter] = np.maximum(ttf_roi_kpt_offset[index, jter], fake_heatmap)

                    # float coordinate
                    inds = fake_heatmap < ttf_roi_kpt_offset[index, jter]
                    fake_heatmap[inds] = 0
                    inds = fake_heatmap > 0
                    mm = gt_kpts_reg[index]
                    mm[:, inds] = kpt_pt[:, None]
                    gt_kpts_reg[index] = mm
                
                gt_kpts_reg_weight[index, 0] = np.max(ttf_roi_kpt_offset[index], 0)

                # image = sample['image'].copy()
                # image = np.transpose(image, (1,2,0)).astype(np.uint8).copy()
                # human_x0 = np.clip((int)(human_x0) * 2, 0, image.shape[1])
                # human_y0 = np.clip((int)(human_y0) * 2, 0, image.shape[0])
                # human_x1 = np.clip((int)(human_x1) * 2, 0, image.shape[1])
                # human_y1 = np.clip((int)(human_y1) * 2, 0, image.shape[0])
                # human_image = image[(human_y0):(human_y1), (human_x0):(human_x1)]
                # cv2.imwrite('./human_image.jpg', human_image)
                # cv2.imwrite(f"./human_kpt.jpg", (np.max(sample['ttf_roi_kpt'][index], 0)*255).astype(np.uint8))

                # x0,y0,x1,y1 = sample['ttf_roi_box'][index]
                # human_x0 = np.clip((int)(x0), 0, image.shape[1])
                # human_y0 = np.clip((int)(y0), 0, image.shape[0])
                # human_x1 = np.clip((int)(x1), 0, image.shape[1])
                # human_y1 = np.clip((int)(y1), 0, image.shape[0])
                # human_image = image[(human_y0):(human_y1), (human_x0):(human_x1)]
                # cv2.imwrite('./human_image2.jpg', human_image)

            sample['ttf_roi_kpt'] = np.clip(sample['ttf_roi_kpt'], a_min=0.0, a_max=1.0)
            sample['ttf_roi_kpt_reg'] = gt_kpts_reg
            sample['ttf_roi_kpt_reg_weight'] = gt_kpts_reg_weight

            if self.offset_generator is not None:
                offset_map, weight_map = self.offset_generator(joints, person_areas)
                sample['ttf_offset_map'] = offset_map
                sample['ttf_offset_weight'] = weight_map            
            if self.paf_generator is not None:
                paf_map = self.paf_generator(joints, person_areas)
                sample['ttf_paf_map']= paf_map
                sample['ttf_paf_weight'] = np.ones(paf_map.shape)
            
            count += 1
            
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, factor=6., scale=0.0, omega=0.3):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / factor * (1 - scale ** omega)
        sigma_y = h / factor * (1 - scale ** omega)
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]+0.5), int(center[1]+0.5)

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap
