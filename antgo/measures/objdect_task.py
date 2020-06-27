# -*- coding: UTF-8 -*-
# @Time    : 17-5-3
# @File    : objdect_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.measures.average_precision import *
from antgo.measures.roc_auc import *
from antgo.measures.precision_recall import *
from antgo.measures.binary_c import *
from antgo.task.task import *
from antgo.measures.base import *
from antgo.utils._bbox import bbox_overlaps
from collections import defaultdict
import antgo.utils._mask as _mask

default = {'AntVOCDet': ('VOC', 'OBJECT-DETECTION'),
           'AntROCandAUCDet': ('ROC_AUC', 'OBJECT-DETECTION'),
           'AntPRDet': ('PR', 'OBJECT-DETECTION'),
           'AntAPRFDet': ('APRF', 'OBJECT-DETECTION'),
           'AntTFTFDet': ('TFTF', 'OBJECT-DETECTION'),
           'AntCOCODet': ('COCO', 'OBJECT-DETECTION'),
           }
class AntVOCDet(AntMeasure):
  def __init__(self, task):
    super(AntVOCDet, self).__init__(task, 'VOC')
    assert(task.task_type == 'OBJECT-DETECTION')
    self.is_support_rank = True

  def eva(self, data, label):
    category_num = len(self.task.class_label)
    detection_score = [[] for _ in range(category_num)]
    detection_label = [[] for _ in range(category_num)]

    if label is not None:
      data = zip(data, label)

    # assert(len(data) == len(label))
    predict_box_total = 0

    samples_scores = []
    # 1.step positive sample is overlap > 0.5
    for predict, gt in data:
      # sample id
      # sample_id = gt['id']

      if predict is None :
        for missed_gt_bi in range(len(gt['bbox'])):
          detection_label[gt['category_id'][missed_gt_bi]].append(1)
          detection_score[gt['category_id'][missed_gt_bi]].append(-float("inf"))
        continue

      det_bbox = predict['det-bbox']
      det_score = predict['det-score']
      det_label = predict['det-label']

      # ground truth bbox and categories
      gt_bbox = np.array(gt['bbox'])
      gt_category = np.array(gt['category_id']).astype(dtype=np.int32)
      gt_bbox_num = gt_bbox.shape[0]

      # predict position, category, and score
      predict_box = np.array(det_bbox)
      predict_category = np.array(det_label).astype(dtype=np.int32)
      predict_score = np.array(det_score)
      predict_box_total += predict_score.shape[0]

      overlaps = bbox_overlaps(
          np.ascontiguousarray(predict_box, dtype=np.float),
          np.ascontiguousarray(gt_bbox, dtype=np.float))
      #overlaps = _mask.iou(predict_box.tolist(), gt_bbox.tolist(), [0 for _ in range(gt_bbox_num)])

      # distinguish positive and negative samples and scores (overlap > 0.5)
      gtm = np.ones((gt_bbox_num)) * (-1)
      for dind, d in enumerate(predict_box.tolist()):
        # information about best match so far (m=-1 -> unmatched)
        m = -1
        iou = 0.5
        for gind, g in enumerate(gt_bbox.tolist()):
          if gt_category[gind] != predict_category[dind]:
            continue

          # if this gt already matched continue
          if gtm[gind] >= 0:
            continue

          # continue to next gt unless better match made
          if overlaps[dind, gind] < iou:
            continue
          # if match successful and best so far, store appropriately
          iou = overlaps[dind, gind]
          m = gind

        # if match made store id of match for both dt and gt
        if m > -1:
          gtm[m] = dind

          # success to match
          detection_score[gt_category[m]].append(predict_score[dind])
          detection_label[gt_category[m]].append(1)

          # # record sample
          # samples_scores.append({'id': sample_id,
          #                        'score': 1,
          #                        'category': gt_category[m],
          #                        'box': gt_bbox[m].tolist(),
          #                        'index': sample_id * 100 + m})

      # process none matched det
      for dind, d in enumerate(predict_box.tolist()):
        if dind not in gtm:
          detection_score[predict_category[dind]].append(predict_score[dind])
          detection_label[predict_category[dind]].append(0)

      # process missed gt bbox
      missed_gt_bbox = [_ for _ in range(gt_bbox_num) if gtm[_] == -1]
      for missed_gt_bi in missed_gt_bbox:
        detection_label[gt_category[missed_gt_bi]].append(1)
        detection_score[gt_category[missed_gt_bi]].append(-float("inf"))

        # # record sample
        # samples_scores.append({'id': sample_id,
        #                        'score': 0,
        #                        'category': gt_category[missed_gt_bi],
        #                        'box': gt_bbox[missed_gt_bi].tolist(),
        #                        'index': sample_id * 100 + missed_gt_bi})
      #########################################################################

    # 2.step compute mean average precision
    voc_mean_map = []
    for predict, gt in zip(detection_label, detection_score):
      result = vmap(predict, gt)
      if result is None:
        result = 0.0
      voc_mean_map.append(float(result))

    # 3.step make json
    voc_map = float(np.mean(voc_mean_map))
    return {'statistic': {'name': self.name,
                          'value': [{'name': 'Mean-MAP', 'value': voc_map, 'type': 'SCALAR'},
                                    {'name': 'MAP', 'value': voc_mean_map, 'type': 'SCALAR', 'x': 'class', 'y': 'Mean Average Precision'}]},
            'info': samples_scores}


class AntROCandAUCDet(AntMeasure):
    def __init__(self, task):
        super(AntROCandAUCDet, self).__init__(task, 'ROC_AUC')
        assert(task.task_type == 'OBJECT-DETECTION')

    def eva(self, data, label):
        category_num = len(self.task.class_label)
        detection_score = [[] for _ in range(category_num)]
        detection_label = [[] for _ in range(category_num)]

        if label is not None:
            data = zip(data, label)

        # assert(len(data) == len(label))
        #
        overlap_thre = float(getattr(self.task, 'roc_auc_overlap', 0.5))

        # 1.step positive sample is overlap > overlap_thre
        predict_box_total = 0
        for predict, gt in data:
            if predict is None:
                for missed_gt_bi in range(len(gt['bbox'])):
                    detection_label[gt['category_id'][missed_gt_bi]].append(1)
                    detection_score[gt['category_id'][missed_gt_bi]].append(-float("inf"))
                continue

            det_bbox = predict['det-bbox']
            det_score = predict['det-score']
            det_label = predict['det-label']

            # ground truth bbox and categories
            gt_bbox = np.array(gt['bbox'])
            gt_category = np.array(gt['category_id']).astype(dtype=np.int32)
            gt_bbox_num = gt_bbox.shape[0]

            # predict position, category, and score
            predict_box = np.array(det_bbox)
            predict_category = np.array(det_label).astype(dtype=np.int32)
            predict_score = np.array(det_score)
            predict_box_total += predict_score.shape[0]

            overlaps = bbox_overlaps(
                np.ascontiguousarray(predict_box, dtype=np.float),
                np.ascontiguousarray(gt_bbox, dtype=np.float))

            # distinguish positive and negative samples and scores (overlap > 0.5)
            gtm = np.ones((gt_bbox_num)) * (-1)
            for dind, d in enumerate(predict_box.tolist()):
                # information about best match so far (m=-1 -> unmatched)
                m = -1
                iou = 0.5
                for gind, g in enumerate(gt_bbox.tolist()):
                    if gt_category[gind] != predict_category[dind]:
                        continue

                    # if this gt already matched  continue
                    if gtm[gind] >= 0:
                        continue

                    # continue to next gt unless better match made
                    if overlaps[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = overlaps[dind, gind]
                    m = gind

                # if match made store id of match for both dt and gt
                if m > -1:
                    gtm[m] = dind

                    # success to match
                    detection_score[gt_category[m]].append(predict_score[dind])
                    detection_label[gt_category[m]].append(1)

            # process none matched det
            for dind, d in enumerate(predict_box.tolist()):
                if dind not in gtm:
                    detection_score[predict_category[dind]].append(predict_score[dind])
                    detection_label[predict_category[dind]].append(0)

            # process missed gt bbox
            missed_gt_bbox = [_ for _ in range(gt_bbox_num) if gtm[_] == -1]
            for missed_gt_bi in missed_gt_bbox:
                detection_label[gt_category[missed_gt_bi]].append(1)
                detection_score[gt_category[missed_gt_bi]].append(-float("inf"))

            #########################################################################

        # 2.step compute ROC curve and AUC
        category_roc_curves = []
        category_auc_scroes = []
        for predict, gt in zip(detection_label, detection_score):
            # skip 0
          
            roc_curve = roc(predict, gt)
            if roc_curve is None:
                roc_curve = np.array(())
            auc_score = auc(predict, gt)
            if auc_score is None:
                auc_score = 0
            category_roc_curves.append(roc_curve.tolist())
            category_auc_scroes.append(float(auc_score))

        return {'statistic': {'name': self.name,
                              'value': [{'name': 'ROC',
                                         'value': category_roc_curves,
                                         'type': 'CURVE',
                                         'x': 'FP',
                                         'y': 'TP'},
                                        {'name': 'AUC',
                                         'value': category_auc_scroes,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'AUC'}]},}


class AntPRDet(AntMeasure):
    def __init__(self, task):
        super(AntPRDet, self).__init__(task, 'PR')
        assert(task.task_type == 'OBJECT-DETECTION')

    def eva(self, data, label):
        category_num = len(self.task.class_label)
        detection_score = [[] for _ in range(category_num)]
        detection_label = [[] for _ in range(category_num)]

        if label is not None:
            data = zip(data, label)

        # assert(len(data) == len(label))

        #
        overlap_thre = float(getattr(self.task, 'pr_overlap', 0.5))

        # 1.step positive sample is overlap > overlap_thre
        predict_box_total = 0
        for predict, gt in data:
            if predict is None:
                for missed_gt_bi in range(len(gt['bbox'])):
                    detection_label[gt['category_id'][missed_gt_bi]].append(1)
                    detection_score[gt['category_id'][missed_gt_bi]].append(-float("inf"))
                continue

            det_bbox = predict['det-bbox']
            det_score = predict['det-score']
            det_label = predict['det-label']

            # ground truth bbox and categories
            gt_bbox = np.array(gt['bbox'])
            gt_category = np.array(gt['category_id']).astype(dtype=np.int32)
            gt_bbox_num = gt_bbox.shape[0]

            # predict position, category, and score
            predict_box = np.array(det_bbox)
            predict_category = np.array(det_label).astype(dtype=np.int32)
            predict_score = np.array(det_score)
            predict_box_total += predict_score.shape[0]

            overlaps = bbox_overlaps(
                np.ascontiguousarray(predict_box, dtype=np.float),
                np.ascontiguousarray(gt_bbox, dtype=np.float))

            # distinguish positive and negative samples and scores (overlap > 0.5)
            gtm = np.ones((gt_bbox_num)) * (-1)
            for dind, d in enumerate(predict_box.tolist()):
                # information about best match so far (m=-1 -> unmatched)
                m = -1
                iou = 0.5
                for gind, g in enumerate(gt_bbox.tolist()):
                    if gt_category[gind] != predict_category[dind]:
                        continue

                    # if this gt already matched  continue
                    if gtm[gind] >= 0:
                        continue

                    # continue to next gt unless better match made
                    if overlaps[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = overlaps[dind, gind]
                    m = gind

                # if match made store id of match for both dt and gt
                if m > -1:
                    gtm[m] = dind

                    # success to match
                    detection_score[gt_category[m]].append(predict_score[dind])
                    detection_label[gt_category[m]].append(1)

            # process none matched det
            for dind, d in enumerate(predict_box.tolist()):
                if dind not in gtm:
                    detection_score[predict_category[dind]].append(predict_score[dind])
                    detection_label[predict_category[dind]].append(0)

            # process missed gt bbox
            missed_gt_bbox = [_ for _ in range(gt_bbox_num) if gtm[_] == -1]
            for missed_gt_bi in missed_gt_bbox:
                detection_label[gt_category[missed_gt_bi]].append(1)
                detection_score[gt_category[missed_gt_bi]].append(-float("inf"))

            #########################################################################

        # 2.step compute Precicion Recall curve
        category_pr_curves = []
        for predict, gt in zip(detection_label, detection_score):
            # skip 0
          
            pr_curve = pr(predict, gt)
            if pr_curve is None:
                pr_curve = np.array(())
            category_pr_curves.append(pr_curve.tolist())

        return {'statistic': {'name': self.name,
                              'value': [{'name': 'Precision-Recall',
                                         'value': category_pr_curves,
                                         'type': 'CURVE',
                                         'x': 'recall',
                                         'y': 'precision'}]},}


class AntAPRFDet(AntMeasure):
    def __init__(self, task):
        super(AntAPRFDet, self).__init__(task, 'APRF')
        assert(task.task_type == 'OBJECT-DETECTION')

    def eva(self, data, label):
        category_num = len(self.task.class_label)
        detection_score = [[] for _ in range(category_num)]
        detection_label = [[] for _ in range(category_num)]

        if label is not None:
            data = zip(data, label)

        # assert(len(data) == len(label))

        #
        overlap_thre = float(getattr(self.task, 'aprf_overlap', 0.5))

        # 1.step positive sample is overlap > overlap_thre
        predict_box_total = 0
        for predict, gt in data:
            if predict is None:
                for missed_gt_bi in range(len(gt['bbox'])):
                    detection_label[gt['category_id'][missed_gt_bi]].append(1)
                    detection_score[gt['category_id'][missed_gt_bi]].append(-float("inf"))
                continue

            det_bbox = predict['det-bbox']
            det_score = predict['det-score']
            det_label = predict['det-label']

            # ground truth bbox and categories
            gt_bbox = np.array(gt['bbox'])
            gt_category = np.array(gt['category_id']).astype(dtype=np.int32)
            gt_bbox_num = gt_bbox.shape[0]

            # predict position, category, and score
            predict_box = np.array(det_bbox)
            predict_category = np.array(det_label).astype(dtype=np.int32)
            predict_score = np.array(det_score)
            predict_box_total += predict_score.shape[0]

            overlaps = bbox_overlaps(
                np.ascontiguousarray(predict_box, dtype=np.float),
                np.ascontiguousarray(gt_bbox, dtype=np.float))

            # distinguish positive and negative samples and scores (overlap > 0.5)
            gtm = np.ones((gt_bbox_num)) * (-1)
            for dind, d in enumerate(predict_box.tolist()):
                # information about best match so far (m=-1 -> unmatched)
                m = -1
                iou = 0.5
                for gind, g in enumerate(gt_bbox.tolist()):
                    if gt_category[gind] != predict_category[dind]:
                        continue

                    # if this gt already matched  continue
                    if gtm[gind] >= 0:
                        continue

                    # continue to next gt unless better match made
                    if overlaps[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = overlaps[dind, gind]
                    m = gind

                # if match made store id of match for both dt and gt
                if m > -1:
                    gtm[m] = dind

                    # success to match
                    detection_score[gt_category[m]].append(predict_score[dind])
                    detection_label[gt_category[m]].append(1)

            # process none matched det
            for dind, d in enumerate(predict_box.tolist()):
                if dind not in gtm:
                    detection_score[predict_category[dind]].append(predict_score[dind])
                    detection_label[predict_category[dind]].append(0)

            # process missed gt bbox
            missed_gt_bbox = [_ for _ in range(gt_bbox_num) if gtm[_] == -1]
            for missed_gt_bi in missed_gt_bbox:
                detection_label[gt_category[missed_gt_bi]].append(1)
                detection_score[gt_category[missed_gt_bi]].append(-float("inf"))

            #########################################################################

        # 2.step compute accuracy, precision, recall, and F1
        category_accuracy = []
        category_precision = []
        category_recall = []
        category_F1 = []
        for category_detection_label, category_detection_score in zip(detection_label, detection_score):
            # skip 0
          
            predicated_label = [1 if s > -float("inf") else 0 for s in category_detection_score]

            res = binary_c_stats2(actual=category_detection_label, predicated=predicated_label)
            accuracy, precision, recall, F1 = res if res is not None else 0, 0, 0, 0

            category_accuracy.append(accuracy)
            category_precision.append(precision)
            category_recall.append(recall)
            category_F1.append(F1)

        return {'statistic': {'name': self.name,
                              'value': [{'name': 'Accuracy',
                                         'value': category_accuracy,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'accuracy'},
                                        {'name': 'Precision',
                                         'value': category_precision,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'precision'},
                                        {'name': 'Recall',
                                         'value': category_recall,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'recall'},
                                        {'name': 'F1',
                                         'value': category_F1,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'F1'}]}}


class AntTFTFDet(AntMeasure):
    def __init__(self, task):
        super(AntTFTFDet, self).__init__(task, 'TFTF')
        assert(task.task_type == 'OBJECT-DETECTION')

    def eva(self, data, label):
        category_num = len(self.task.class_label)
        detection_score = [[] for _ in range(category_num)]
        detection_label = [[] for _ in range(category_num)]

        if label is not None:
            data = zip(data, label)
        # assert(len(data) == len(label))

        #
        overlap_thre = float(getattr(self.task,'APRF_overlap',0.5))

        # 1.step positive sample is overlap > overlap_thre
        predict_box_total = 0
        for predict, gt in data:
            if predict is None:
                for missed_gt_bi in range(len(gt['bbox'])):
                    detection_label[gt['category_id'][missed_gt_bi]].append(1)
                    detection_score[gt['category_id'][missed_gt_bi]].append(-float("inf"))
                continue

            det_bbox = predict['det-bbox']
            det_score = predict['det-score'].flatten()
            det_label = predict['det-label'].flatten()

            # ground truth bbox and categories
            gt_bbox = np.array(gt['bbox'])
            gt_category = np.array(gt['category_id']).astype(dtype=np.int32)
            gt_bbox_num = gt_bbox.shape[0]

            # predict position, category, and score
            predict_box = np.array(det_bbox)
            predict_category = np.array(det_label).astype(dtype=np.int32)
            predict_score = np.array(det_score)
            predict_box_total += predict_score.shape[0]

            overlaps = bbox_overlaps(
                np.ascontiguousarray(predict_box, dtype=np.float),
                np.ascontiguousarray(gt_bbox, dtype=np.float))

            # distinguish positive and negative samples and scores (overlap > 0.5)
            gtm = np.ones((gt_bbox_num)) * (-1)
            for dind, d in enumerate(predict_box.tolist()):
                # information about best match so far (m=-1 -> unmatched)
                m = -1
                iou = 0.5
                for gind, g in enumerate(gt_bbox.tolist()):
                    if gt_category[gind] != predict_category[dind]:
                        continue

                    # if this gt already matched  continue
                    if gtm[gind] >= 0:
                        continue

                    # continue to next gt unless better match made
                    if overlaps[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = overlaps[dind, gind]
                    m = gind

                # if match made store id of match for both dt and gt
                if m > -1:
                    gtm[m] = dind

                    # success to match
                    detection_score[gt_category[m]].append(predict_score[dind])
                    detection_label[gt_category[m]].append(1)

            # process none matched det
            for dind, d in enumerate(predict_box.tolist()):
                if dind not in gtm:
                    detection_score[predict_category[dind]].append(predict_score[dind])
                    detection_label[predict_category[dind]].append(0)

            # process missed gt bbox
            missed_gt_bbox = [_ for _ in range(gt_bbox_num) if gtm[_] == -1]
            for missed_gt_bi in missed_gt_bbox:
                detection_label[gt_category[missed_gt_bi]].append(1)
                detection_score[gt_category[missed_gt_bi]].append(-float("inf"))

            #########################################################################

        # 2.step compute true positive(TP), false negative(FN), true negative(TN), false positive(FP)
        category_TP = []
        category_FN = []
        category_TN = []
        category_FP = []
        for category_detection_label,category_detection_score in zip(detection_label,detection_score):
            # skip 0
            predicated_label = [1 if s > -float("inf") else 0 for s in category_detection_score]

            res = binary_c_stats(actual=category_detection_label, predicated=predicated_label)
            TP,FN,TN,FP = res if res is not None else 0, 0, 0, 0

            category_TP.append(TP)
            category_FN.append(FN)
            category_TN.append(TN)
            category_FP.append(FP)

        return {'statistic': {'name': self.name,
                              'value': [{'name': 'true-positive',
                                         'value': category_TP,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'TP'},
                                        {'name': 'false-negative',
                                         'value': category_FN,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'FN'},
                                        {'name': 'true-negative',
                                         'value': category_TN,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'TN'},
                                        {'name': 'false-positive',
                                         'value': category_FP,
                                         'type': 'SCALAR',
                                         'x': 'class',
                                         'y': 'FP'}]},}


class AntCOCODet(AntMeasure):
    def __init__(self, task):
        super(AntCOCODet, self).__init__(task, 'COCO')
        assert(task.task_type == 'OBJECT-DETECTION')

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1
        self.dts = []
        self.gts = []
        self.imgIds = []
        self.catIds = []

    def _parse_result(self, data, label):
        if label is not None:
            data = zip(data, label)

        # assert(len(data) == len(label))
        gt_id = 1
        det_id = 1
        for index, (a, b) in enumerate(data):
            image_id = -1
            for gt_box_i, gt_box in enumerate(b['bbox']):
                gt = {}
                x0, y0, x1, y1 = gt_box
                gt['bbox'] = np.array([x0, y0, x1, y1])
                gt['area'] = b['area'][gt_box_i]
                gt['image_id'] = b['id']
                gt['category_id'] = b['category_id'][gt_box_i]
                gt['iscrowd'] = b['iscrowd'][gt_box_i] if 'iscrowd' in b else 0
                gt['id'] = gt_id
                gt_id += 1
                image_id = b['id']
                self.imgIds.append(image_id)
                self.catIds.append(b['category_id'][gt_box_i])
                self.gts.append(gt)

            if a is None:
                dt = {}
                dt['bbox'] = None
                dt['area'] = 0
                dt['image_id'] = -1
                dt['category_id'] = -1
                dt['score'] = -1
                dt['id'] = -1
                self.dts.append(dt)
                continue

            # a dict {'det-bbox':, 'det-score':, 'det-label':,}
            # b annotation
            det_bbox = a['det-bbox']
            det_score = a['det-score'].flatten()
            det_label = a['det-label'].flatten()

            for box_i, box in enumerate(det_bbox):
                dt = {}
                x0, y0, x1, y1 = box
                score = det_score[box_i]
                label = int(det_label[box_i])

                dt['bbox'] = np.array([x0, y0, x1, y1])
                dt['area'] = (x1 - x0) * (y1 - y0)
                dt['image_id'] = image_id
                dt['category_id'] = int(label)
                dt['score'] = score
                dt['id'] = det_id
                det_id += 1

                self.dts.append(dt)

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        # set ignore flag
        for gt in self.gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in self.gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in self.dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def _evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        #p = self.params

        self.imgIds = list(np.unique(self.imgIds))
        self.catIds = list(np.unique(self.catIds))
        self.maxDets = sorted(self.maxDets)

        self._prepare()
        # loop through images, area range, max detection number
        catIds = self.catIds
        computeIoU = self._computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in self.imgIds
                        for catId in catIds}

        maxDet = self.maxDets[-1]
        self.evalImgs = [self._evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in self.areaRng
                 for imgId in self.imgIds
             ]

    def _computeIoU(self, imgId, catId):
        # p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets[-1]:
            dt = dt[0:self.maxDets[-1]]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = _mask.iou(d, g, iscrowd)
        return ious

    def _evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]

        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(self.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(self.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind]  = gt[m]['id']
                    gtm[tind, m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def _accumulate(self):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        # if p is None:
        #     p = self.params

        T           = len(self.iouThrs)
        R           = len(self.recThrs)
        K           = len(self.catIds)
        A           = len(self.areaRng)
        M           = len(self.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))

        # create dictionary for future indexing
        catIds = self.catIds
        setK = set(catIds)
        setA = set(map(tuple, self.areaRng))
        setM = set(self.maxDets)
        setI = set(self.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(self.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(self.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), self.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(self.imgIds)  if i in setI]
        I0 = len(self.imgIds)
        A0 = len(self.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, self.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                if pi == len(pr):
                                    q[ri] = pr[pi - 1]
                                else:
                                    q[ri] = pr[pi]
                        except:
                            pass

                        precision[t,:,k,a,m] = np.array(q)
        self.eval = {
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall':   recall,
        }

    def eva(self, data, label):
        self._parse_result(data, label)
        self._evaluate()
        self._accumulate()

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(self.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])

            return mean_s, iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)

        # all
        coco_status, coco_status_str = _summarize(1)
        # iou-thre = .5, maxdets = 100
        coco_iou05_maxdets100, coco_iou05_maxdets100_str = \
            _summarize(1, iouThr=.5, maxDets=self.maxDets[2])
        # iou-thre = .75, maxdets = 100
        coco_iou075_maxdets100, coco_iou075_maxdets100_str = \
            _summarize(1, iouThr=.75, maxDets=self.maxDets[2])
        # area-small, maxdets = 100
        coco_small_maxdets100, coco_small_maxdets100_str = \
            _summarize(1, areaRng='small', maxDets=self.maxDets[2])
        # area-medium, maxdets = 100
        coco_medium_maxdets100, coco_medium_maxdets100_str = \
            _summarize(1, areaRng='medium', maxDets=self.maxDets[2])
        # area-large, maxdets = 100
        coco_large_maxdets100, coco_large_maxdets100_str = \
            _summarize(1, areaRng='large', maxDets=self.maxDets[2])
        # maxdets = 1
        coco_maxdets1, coco_maxdets1_str = \
            _summarize(0, maxDets=self.maxDets[0])
        # maxdets = 10
        coco_maxdets10, coco_maxdets10_str = \
            _summarize(0, maxDets=self.maxDets[1])
        # maxdets = 100
        coco_maxdets100, coco_maxdets100_str = \
            _summarize(0, maxDets=self.maxDets[2])
        # area-small, maxdets = 100
        coco_small_maxdets100_nap, coco_small_maxdets100_nap_str = \
            _summarize(0, areaRng='small', maxDets=self.maxDets[2])
        # area-medium, maxdets = 100
        coco_medium_maxdets100_nap, coco_medium_maxdets100_nap_str = \
            _summarize(0, areaRng='medium', maxDets=self.maxDets[2])
        # area-large, maxdets = 100
        coco_large_maxdets100_nap, coco_large_maxdets100_nap_str = \
            _summarize(0, areaRng='large', maxDets=self.maxDets[2])

        return {'statistic':{'name':self.name,
                             'value':[{'name': self.name,
                                       'value': coco_status,
                                       'type': 'SCALAR',
                                       'x': coco_status_str},
                                      {'name': 'iou-0.5-max-det-100-AP',
                                       'value': coco_iou05_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_iou05_maxdets100_str},
                                      {'name': 'iou-0.75-max-det-100-AP',
                                       'value': coco_iou075_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_iou075_maxdets100_str},
                                      {'name': 'area-small-AP',
                                       'value': coco_small_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_small_maxdets100_str},
                                      {'name': 'area-medium-AP',
                                       'value': coco_medium_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_medium_maxdets100_str},
                                      {'name': 'area-large-AP',
                                       'value': coco_large_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_large_maxdets100_str},
                                      {'name': 'max-det-1',
                                       'value': coco_maxdets1,
                                       'type': 'SCALAR',
                                       'x': coco_maxdets1_str},
                                      {'name': 'max-det-10',
                                       'value': coco_maxdets10,
                                       'type': 'SCALAR',
                                       'x': coco_maxdets10_str},
                                      {'name': 'max-det-100',
                                       'value': coco_maxdets100,
                                       'type': 'SCALAR',
                                       'x': coco_maxdets100_str},
                                      {'name': 'area-small',
                                       'value': coco_small_maxdets100_nap,
                                       'type': 'SCALAR',
                                       'x': coco_small_maxdets100_nap_str},
                                      {'name': 'area-medium',
                                       'value': coco_medium_maxdets100_nap,
                                       'type': 'SCALAR',
                                       'x': coco_medium_maxdets100_nap_str},
                                      {'name': 'area-large',
                                       'value': coco_large_maxdets100_nap,
                                       'type': 'SCALAR',
                                       'x': coco_large_maxdets100_nap_str}]},}