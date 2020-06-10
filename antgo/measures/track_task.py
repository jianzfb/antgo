# -*- coding: UTF-8 -*-
# @Time    : 2019-04-17 09:38
# @File    : track_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.base import *
from antgo.utils._bbox import bbox_overlaps
from antgo.task.task import *

default={'AntTrackStability': ('TrackStability', 'TRACK')}


class AntTrackStability(AntMeasure):
  def __init__(self, task):
    super(AntTrackStability, self).__init__(task, "TrackStability")
    assert(task.task_type == 'TRACK')
    self.iou_thres = 0.5
    self.support_rank_index = 0
    self.is_support_rank = True
    self.is_inverse = True

  def _fragment_error_fast(self, trajectory):
    t_k = len(trajectory)
    if t_k <= 1:
      return [0.0] * 11

    f_k = np.zeros((len(trajectory)))
    scores_k = np.zeros((len(trajectory)))

    index = 0
    for det_bbox, gt_bbox, det_score, is_ok in trajectory:
      scores_k[index] = det_score
      if index == 0:
        f_k[index] = 1.0 if is_ok else 0
      else:
        f_k[index] = f_k[index-1]+1 if is_ok else f_k[index-1]
      index += 1

    sampling_p = np.array(range(11)) * 0.1
    error_k = f_k / (t_k - 1.0)
    error_k = np.interp(sampling_p, scores_k.tolist()[::-1], error_k.tolist()[::-1])
    return error_k

  def _center_position_error_fast(self, trajectory):
    if len(trajectory) <= 1:
      return [0.0] * 11

    e_x = np.zeros((len(trajectory)))
    e_y = np.zeros((len(trajectory)))
    error_k = np.zeros((len(trajectory) - 1))
    scores_k = np.zeros((len(trajectory) - 1))
    index = 0
    for det_bbox, gt_bbox, det_socre, is_ok in trajectory:
      if not is_ok:
        if index > 1:
          error_k[index - 1] = error_k[index - 2]
          scores_k[index - 1] = det_socre
          index += 1
        continue

      e_x[index] = (det_bbox[0] - gt_bbox[0])/(gt_bbox[2]-gt_bbox[0])
      e_y[index] = (det_bbox[1] - gt_bbox[1])/(gt_bbox[3]-gt_bbox[1])
      if index > 0:
        error_k[index-1] = np.std(e_x[0:index+1])+np.std(e_y[0:index+1])
        scores_k[index-1] = det_socre
      index += 1

    sampling_p = np.array(range(11)) * 0.1
    error_k = np.interp(sampling_p, scores_k.tolist()[::-1], error_k.tolist()[::-1])
    return error_k

  def _scale_ratio_error_fast(self, trajectory):
    if len(trajectory) <= 1:
      return [0.0] * 11

    e_s = np.zeros((len(trajectory)))
    e_r = np.zeros((len(trajectory)))
    error_k = np.zeros((len(trajectory) - 1))
    scores_k = np.zeros((len(trajectory) - 1))
    index = 0
    for det_bbox, gt_bbox, det_socre, is_ok in trajectory:
      if not is_ok:
        if index > 1:
          error_k[index - 1] = error_k[index - 2]
          scores_k[index - 1] = det_socre
          index += 1
        continue

      e_s[index] = np.sqrt(((det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])) / (
              (gt_bbox[2]-gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])))

      e_r[index] = ((det_bbox[2] - det_bbox[0]) / (det_bbox[3] - det_bbox[1])) / (
                (gt_bbox[2] - gt_bbox[0]) / (gt_bbox[3] - gt_bbox[1]))

      if index > 0:
        error_k[index - 1] = np.std(e_s[0:index+1]) + np.std(e_r[0:index+1])
      scores_k[index - 1] = det_socre
      index += 1

    sampling_p = np.array(range(11)) * 0.1
    error_k = np.interp(sampling_p, scores_k.tolist()[::-1], error_k.tolist()[::-1])
    return error_k

  def eva(self, data, label):
    # video list
    if label is not None:
      data = zip(data, label)

    all_trajectory_list = []
    for video_predict, video_gt in data:
      trajectory_list = []
      for det_predict, det_gt in zip(video_predict, video_gt):
        # predict bbox in frame
        det_bbox = np.array(det_predict['det-bbox'])
        det_socre = np.array(det_predict['det-score'])
        det_label = np.array(det_predict['det-label']).astype(dtype=np.int32)

        # gt bbox in frame
        gt_bbox = det_gt['bbox']
        gt_category = det_gt['category_id']
        gt_bbox_num = len(gt_bbox)
        if len(trajectory_list) == 0:
          trajectory_list = [[] for _ in range(gt_bbox_num)]

        overlaps = bbox_overlaps(
          np.ascontiguousarray(det_bbox, dtype=np.float),
          np.ascontiguousarray(gt_bbox, dtype=np.float))

        gtm = np.ones((gt_bbox_num)) * (-1)
        for dind, d in enumerate(det_bbox.tolist()):
          best_match = -1
          iou_thres = self.iou_thres
          for gind, g in enumerate(gt_bbox):
            if int(gt_category[gind]) != int(det_label[dind]):
              continue

            if gtm[gind] >= 0:
              continue

            if overlaps[dind, gind] < iou_thres:
              continue

            iou_thres = overlaps[dind, gind]
            best_match = gind

          if best_match > -1:
            gtm[best_match] = dind
            trajectory_list[int(gt_category[best_match])].append((det_bbox[dind],
                                                                  gt_bbox[best_match],
                                                                  det_socre[dind],
                                                                  True))
        for gind, g in enumerate(gt_bbox):
            if gtm[gind] == -1:
              trajectory_list[int(gt_category[gind])].append(([],
                                                              gt_bbox[gind],
                                                              0.0,
                                                              False))

      all_trajectory_list.extend(trajectory_list)

    trajectory_error_list = []
    trajectory_f_error_list = []
    trajectory_cp_error_list = []
    trajectory_sr_error_list = []
    for trajectory in all_trajectory_list:
      ordered_trajectory = sorted(trajectory, key=lambda x: x[2], reverse=True)

      error_f = self._fragment_error_fast(ordered_trajectory)
      error_cpe = self._center_position_error_fast(ordered_trajectory)
      error_sre = self._scale_ratio_error_fast(ordered_trajectory)
      trajectory_f_error_list.append(error_f.tolist())
      trajectory_cp_error_list.append(error_cpe.tolist())
      trajectory_sr_error_list.append(error_sre.tolist())
      trajectory_error_list.append((error_f+error_cpe+error_sre).tolist())

    trajectory_error_mean = np.mean(trajectory_error_list, 0)
    trajectory_f_error_mean = np.mean(trajectory_f_error_list, 0)
    trajectory_cp_error_mean = np.mean(trajectory_cp_error_list, 0)
    trajectory_sr_error_mean = np.mean(trajectory_sr_error_list, 0)
    stability_e = np.sum(trajectory_error_mean)

    return {'statistic': {'name': self.name,
                          'value': [{'name': 'TRAJECTORY_STABILITY',
                                     'value': stability_e,
                                     'type': 'SCALAR',
                                     'x': 'class',
                                     'y': 'STABILITY'},
                                    {'name': 'TRAJECTORY_FRAGMENT_ERROR',
                                     'value': trajectory_f_error_mean.tolist(),
                                     'type': 'SCALAR',
                                     'x': 'Detect Threshold',
                                     'y': 'Fragment Error'},
                                    {'name': 'TRAJECTORY_CENTER_POSITION_ERROR',
                                     'value': trajectory_cp_error_mean.tolist(),
                                     'type': 'SCALAR',
                                     'x': 'Detect Threshold',
                                     'y': 'Center Position Error'},
                                    {'name': 'TRAJECTORY_SCALE_RATIO_ERROR',
                                     'value': trajectory_sr_error_mean.tolist(),
                                     'type': 'SCALAR',
                                     'x': 'Detect Threshold',
                                     'y': 'Scale Ratio Error'},
                                    {'name': 'TRAJECTORY_ERROR',
                                      'value': trajectory_error_mean.tolist(),
                                     'type': 'SCALAR',
                                     'x': 'Detect Threshold',
                                     'y': 'Total Error'}]}}


if __name__ == '__main__':
  data = []
  video_clips_num = 10
  for _ in range(video_clips_num):
    det_result = []
    gt_result = []
    for frame_index in range(30):
      # target 1
      x1 = np.random.random()
      y1 = np.random.random()

      x2 = x1 + np.random.random()
      y2 = y1 + np.random.random()

      gt_x1 = x1 + 0.01
      gt_y1 = y1 + 0.1
      gt_x2 = gt_x1 + 0.1
      gt_y2 = gt_y1 + 0.1

      # target 2
      second_x1 = np.random.random()
      second_y1 = np.random.random()

      second_x2 = second_x1 + np.random.random()
      second_y2 = second_y1 + np.random.random()

      gt_second_x1 = second_x1 + 0.01
      gt_second_y1 = second_y1 + 0.1
      gt_second_x2 = gt_second_x1 + 0.1
      gt_second_y2 = gt_second_y1 + 0.2

      det_result.append({'det-bbox': [[x1, y1, x2, y2], [second_x1, second_y1, second_x2, second_y2]],
                         'det-score': [np.random.random(), np.random.random()],
                         'det-label': [0, 1]})

      gt_result.append({'bbox': [[gt_x1, gt_y1, gt_x2, gt_y2],
                                 [gt_second_x1, gt_second_y1, gt_second_x2, gt_second_y2]],
                        'category_id': [0, 1]})

    data.append([det_result, gt_result])

  ts = AntTrackStability(None)
  result = ts.eva(data, None)