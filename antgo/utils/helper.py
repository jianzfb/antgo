from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr
from antgo.utils._bbox import bbox_overlaps

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # size = w * h
    # size_ratios = size / ratios
    #
    # ws = np.round(np.sqrt(size_ratios))
    # hs = np.round(ws * ratios)
    ws = np.round(w * np.power(ratios,0.5))
    hs = np.round(h * np.power(ratios,-0.5))
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def make_anchors(base_size=16, ratios=[0.5, 1, 2],
                 scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], np.array(scales))
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def make_shift_anchors(height,width,stride,anchors):
    # Enumerate all shifts
    shift_x = np.arange(0, width) * stride
    shift_y = np.arange(0, height) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    shift_anchors = anchors.reshape((1, A, 4)) + \
                      shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    shift_anchors = shift_anchors.reshape((K * A, 4))

    return shift_anchors


def bbox_transform(proposals, gt_rois):
    assert(proposals.shape[0] == gt_rois.shape[0])
    ex_widths = proposals[:, 2] - proposals[:, 0] + 1.0
    ex_heights = proposals[:, 3] - proposals[:, 1] + 1.0
    ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(default_boxes, deltas):
    # support batch data (for deltas)
    if default_boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    default_boxes = default_boxes.astype(deltas.dtype, copy=True)
    default_boxes = np.tile(default_boxes, [deltas.shape[0] // default_boxes.shape[0], 1])

    widths = default_boxes[:, 2] - default_boxes[:, 0] + 1.0
    heights = default_boxes[:, 3] - default_boxes[:, 1] + 1.0
    ctr_x = default_boxes[:, 0] + 0.5 * widths
    ctr_y = default_boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size,max_size = None):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = None
    if max_size is not None:
        keep = np.where((ws >= min_size) & (hs >= min_size) &
                        (ws <= max_size) & (hs <= max_size))[0]
    else:
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def inv_filter_boxes(boxes,min_size,max_size = None):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1

    disable_keep = None
    if max_size is not None:
        disable_keep = np.where((ws < min_size) | (hs < min_size) | 
                                (ws > max_size) | (hs > max_size))[0]
    else:
        disable_keep = np.where((ws < min_size) | (hs < min_size))[0]
    return disable_keep


def batch_positive_and_negative_selecting_strategy(batch_proposals,batch_logits,batch_gt_boxes,batch_gt_labels,neg_pos_ratio=3):
    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    # batch_proposals shape -> (batch_size,proposals_num,4)
    # batch_gt_boxes shape -> (batch_size,bboxes_num,4)
    # batch_gt_labels shape -> (batch_size,bboxes_num)
    proposal_bbox_labels_list = []
    proposal_bbox_targets_list = []
    for batch_index in range(batch_proposals.shape[0]):
        proposals = batch_proposals[batch_index,:,:]
        logits = batch_logits[batch_index,:,:]
        gt_boxes = batch_gt_boxes[batch_index,:,:]
        gt_labels = batch_gt_labels[batch_index,:]

        selected_index = np.where(gt_labels >= 0)
        selected_gt_boxes = gt_boxes[selected_index]
        selected_gt_labels = gt_labels[selected_index]

        proposal_bbox_labels,proposal_bbox_targets = \
            positive_and_negative_selecting_strategy(proposals,logits,selected_gt_boxes,selected_gt_labels,neg_pos_ratio)

        proposal_bbox_labels_list.append(proposal_bbox_labels[np.newaxis,...])
        proposal_bbox_targets_list.append(proposal_bbox_targets[np.newaxis,...])

    batch_proposal_bbox_labels = np.concatenate(proposal_bbox_labels_list,axis=0)
    batch_proposal_bbox_targets = np.concatenate(proposal_bbox_targets_list,axis=0)

    # if len(np.where(batch_proposal_bbox_labels==0)[0]) == 0:
    #     print('no negative samples')
    #
    # if len(np.where(batch_proposal_bbox_labels>=0)[0]) == 0:
    #     print('no proposal data')

    return batch_proposal_bbox_labels,batch_proposal_bbox_targets


def positive_and_negative_selecting_strategy(proposals,logits,gt_boxes,gt_labels,neg_pos_ratio=3,min_size = 5):
    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(proposals, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # max overlap index of each proposal
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(proposals.shape[0]), argmax_overlaps]

    # max overlap index of each gt box
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]

    proposal_labels = np.empty((proposals.shape[0],), dtype=np.int32)
    proposal_labels.fill(-1)
    
    proposal_bbox_index = np.zeros((proposals.shape[0],),dtype=np.int32)
    
    # negative
    proposal_labels[max_overlaps < 0.3] = 0

    # strategy 2 (positive)
    proposal_labels[max_overlaps > 0.5] = gt_labels[argmax_overlaps[max_overlaps > 0.5]]
    proposal_bbox_index[max_overlaps > 0.5] = argmax_overlaps[max_overlaps > 0.5]

    # strategy 1 (positive)
    proposal_labels[gt_argmax_overlaps] = gt_labels
    proposal_bbox_index[gt_argmax_overlaps] = np.arange(0,gt_boxes.shape[0])

    # remove invalid proposals
    disable_keep_index = inv_filter_boxes(proposals,min_size)
    proposal_labels[disable_keep_index] = -1

    # subsample negative labels if we have too many
    fg_inds = np.where(proposal_labels >= 1)[0]
    num_fg = len(fg_inds)
    bg_inds = np.where(proposal_labels == 0)[0]
    if len(bg_inds) > num_fg * neg_pos_ratio:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_fg * neg_pos_ratio), replace=False)
        proposal_labels[disable_inds] = -1

    proposal_bbox_targets = bbox_transform(proposals, gt_boxes[proposal_bbox_index, :])
    proposal_bbox_targets[proposal_labels < 1] = 0

    return proposal_labels, proposal_bbox_targets