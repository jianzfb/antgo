from ..runner.builder import *
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from antgo.dataflow.imgaug.transforms import bbox_xyxy2xywh
from antgo.framework.helper.models.detectors.core.bbox.nms import oks_nms, soft_oks_nms
from .transforms import transform_ann, transform_pred, transform_sigmas
import json


@MEASURES.register_module()
class CocoMetric(object):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        pred_converter (dict, optional): Config dictionary for the prediction
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        gt_converter (dict, optional): Config dictionary for the ground truth
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Defaults to ``'cpu'``
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Defaults to ``None``
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 num_keypoints=33,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 keypoint_sigmas: list=None,
                 outfile_prefix: Optional[str] = None,
                 prefix: Optional[str] = None) -> None:
        super().__init__()
        # self.ann_file = ann_file
        # initialize coco helper with the annotation json file
        # if ann_file is not specified, initialize with the converted dataset
        # if ann_file is not None:
        #     with get_local_path(ann_file) as local_path:
        #         self.coco = COCO(local_path)
        # else:
        #     self.coco = None

        self.coco = None
        self.use_area = use_area
        self.iou_type = iou_type
        self.num_keypoints = num_keypoints
        self.keypoint_sigmas = keypoint_sigmas

        allowed_score_modes = ['bbox', 'bbox_keypoint', 'bbox_rle', 'keypoint']
        if score_mode not in allowed_score_modes:
            raise ValueError(
                "`score_mode` should be one of 'bbox', 'bbox_keypoint', "
                f"'bbox_rle', but got {score_mode}")
        self.score_mode = score_mode
        self.keypoint_score_thr = keypoint_score_thr

        allowed_nms_modes = ['oks_nms', 'soft_oks_nms', 'none']
        if nms_mode not in allowed_nms_modes:
            raise ValueError(
                "`nms_mode` should be one of 'oks_nms', 'soft_oks_nms', "
                f"'none', but got {nms_mode}")
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr

        self.outfile_prefix = outfile_prefix
        self.pred_converter = pred_converter
        self.gt_converter = gt_converter

    def keys(self):
        return {
            'pred': ['pred_img_id', 'pred_keypoint_scores', 'pred_keypoints', 'pred_bbox_scores', 'pred_bbox'], 
            'gt': ['img_id', 'keypoints', 'bbox', 'raw_ann_info', 'image_meta']
        }

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys of the each `gt_dict` in `gt_dicts`:
                    - `img_id`: image id of the data sample
                    - `width`: original image width
                    - `height`: original image height
                    - `raw_ann_info`: the raw annotation information
                Optional keys:
                    - `crowd_index`: measure the crowding level of an image,
                        defined in CrowdPose dataset
                It is worth mentioning that, in order to compute `CocoMetric`,
                there are some required keys in the `raw_ann_info`:
                    - `id`: the id to distinguish different annotations
                    - `image_id`: the image id of this annotation
                    - `category_id`: the category of the instance.
                    - `bbox`: the object bounding box
                    - `keypoints`: the keypoints cooridinates along with their
                        visibilities. Note that it need to be aligned
                        with the official COCO format, e.g., a list with length
                        N * 3, in which N is the number of keypoints. And each
                        triplet represent the [x, y, visible] of the keypoint.
                    - `iscrowd`: indicating whether the annotation is a crowd.
                        It is useful when matching the detection results to
                        the ground truth.
                There are some optional keys as well:
                    - `area`: it is necessary when `self.use_area` is `True`
                    - `num_keypoints`: it is necessary when `self.iou_type`
                        is set as `keypoints_crowd`.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['image_meta']['ori_image_shape'][1],
                    height=gt_dict['image_meta']['ori_image_shape'][0],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                if ann is None:
                    # during evaluation on bottom-up datasets, some images
                    # do not have instance annotation
                    continue

                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    if 'area' not in ann:
                        ann['area'] = (ann['bbox'][2]-ann['bbox'][0]) * (ann['bbox'][3]-ann['bbox'][1])
                    annotation['area'] = ann['area']

                if self.iou_type == 'keypoints_crowd':
                    assert 'num_keypoints' in ann, \
                        '`num_keypoints` is required when `self.iou_type` ' \
                        'is `keypoints_crowd`'
                    annotation['num_keypoints'] = ann['num_keypoints']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmpose CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=[{'supercategory': 'person', 'id': 1, 'name': 'person'}],
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        with open(converted_json_path, 'w') as fp:
            json.dump(coco_json, fp)
        return converted_json_path

    def __call__(self, preds, gts) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            # use converted gt json file to initialize coco helper
            print('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)
        if self.gt_converter is not None:
            for id_, ann in self.coco.anns.items():
                self.coco.anns[id_] = transform_ann(
                    ann, self.gt_converter['num_keypoints'], self.gt_converter['mapping'])

        kpts = defaultdict(list)

        # group the preds by img_id
        instance_id = 0
        for pred in preds:
            img_id = pred['pred_img_id']

            if len(pred['pred_keypoints']) == 0:
                continue
            if self.pred_converter is not None:
                pred = transform_pred(pred, self.pred_converter['num_keypoints'], self.pred_converter['mapping'])

            for idx, keypoints in enumerate(pred['pred_keypoints']):
                instance = {
                    'id': instance_id,
                    'img_id': pred['pred_img_id'],
                    'category_id': pred['pred_category_id'] if 'pred_category_id' in pred else 1,
                    'keypoints': keypoints,
                    'keypoint_scores': pred['pred_keypoint_scores'][idx],
                    'bbox_score': pred['pred_bbox_scores'][idx],
                }
                if 'pred_bbox' in pred:
                    instance['bbox'] = pred['pred_bbox'][idx]

                if 'pred_area' in pred:
                    instance['area'] = pred['pred_area'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)
                instance_id += 1

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts = defaultdict(list)
        if self.pred_converter is not None:
            num_keypoints = self.pred_converter['num_keypoints']
        else:
            num_keypoints = self.num_keypoints

        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate(
                    [
                        instance['keypoints'], instance['keypoint_scores'][:, None]
                    ],
                    axis=-1
                )
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                elif self.score_mode == 'keypoint':
                    instance['score'] = np.mean(instance['keypoint_scores'])
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.keypoint_sigmas)
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # # only format the results without doing quantitative evaluation
        # if self.format_only:
        #     logger.info('results are saved in '
        #                 f'{osp.dirname(outfile_prefix)}')
        #     return {}

        # evaluation results
        eval_results = OrderedDict()
        print(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.num_keypoints
            # collect all the person keypoints in current image
            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = []
            for img_kpt, keypoint in zip(img_kpts, _keypoints):
                res = {
                    'image_id': img_kpt['img_id'],
                    'category_id': img_kpt['category_id'],
                    'keypoints': keypoint.tolist(),
                    'score': float(img_kpt['score']),
                }
                if 'bbox' in img_kpt:
                    res['bbox'] = img_kpt['bbox'].tolist()
                result.append(res)

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'
        # dump(cat_results, res_file, sort_keys=True, indent=4)

        with open(res_file, 'w') as fp:
            json.dump(cat_results, fp)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.keypoint_sigmas
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas,
                             self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == 'keypoints_crowd':
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)',
                'AP(M)', 'AP(H)'
            ]
        else:
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                'AR .75', 'AR (M)', 'AR (L)'
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self,
                                kpts: Dict[int, list],
                                key: str = 'id') -> Dict[int, list]:
        """Sort keypoint detection results in each image and remove the
        duplicate ones. Usually performed in multi-batch testing.

        Args:
            kpts (Dict[int, list]): keypoint prediction results. The keys are
                '`img_id`' and the values are list that may contain
                keypoints of multiple persons. Each element in the list is a
                dict containing the ``'key'`` field.
                See the argument ``key`` for details.
            key (str): The key name in each person prediction results. The
                corresponding value will be used for sorting the results.
                Default: ``'id'``.

        Returns:
            Dict[int, list]: The sorted keypoint detection results.
        """
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts