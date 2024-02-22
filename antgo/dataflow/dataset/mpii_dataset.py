
import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat
from antgo.dataflow.imgaug.transforms import bbox_cs2xyxy
from antgo.dataflow.dataset.base_coco_style_dataset import BaseCocoStyleDataset


__all__ = ['MpiiDataset']
class MpiiDataset(BaseCocoStyleDataset):
    """MPII Dataset for pose estimation.

    "2D Human Pose Estimation: New Benchmark and State of the Art Analysis"
    ,CVPR'2014. More details can be found in the `paper
    <http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf>`__ .

    MPII keypoints::

        0: 'right_ankle'
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        headbox_file (str, optional): The path of ``mpii_gt_val.mat`` which
            provides the headboxes information used for ``PCKh``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/mpii.py')

    def __init__(self,
                 dir: str='',
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 headbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 train_or_test='unkown',
                 max_refetch: int = 1000):

        if headbox_file:
            if data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {data_mode}: '
                    'mode, while "headbox_file" is only '
                    'supported in topdown mode.')

            if not test_mode:
                raise ValueError(
                    f'{self.__class__.__name__} has `test_mode==False` '
                    'while "headbox_file" is only '
                    'supported when `test_mode==True`.')

            headbox_file_type = headbox_file[-3:]
            allow_headbox_file_type = ['mat']
            if headbox_file_type not in allow_headbox_file_type:
                raise KeyError(
                    f'The head boxes file type {headbox_file_type} is not '
                    f'supported. Should be `mat` but got {headbox_file_type}.')
        self.headbox_file = headbox_file

        super().__init__(
            dir=dir,
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
            metainfo=metainfo,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            train_or_test=train_or_test,
            max_refetch=max_refetch)

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in MPII format."""

        local_path = osp.join(self.dir, self.ann_file)
        with open(local_path) as anno_file:
            self.anns = json.load(anno_file)


        if self.headbox_file:
            assert exists(self.headbox_file), 'Headbox file does not exist'
            with get_local_path(self.headbox_file) as local_path:
                self.headbox_dict = loadmat(local_path)
            headboxes_src = np.transpose(self.headbox_dict['headboxes_src'],
                                         [2, 0, 1])
            SC_BIAS = 0.6

        instance_list = []
        image_list = []
        used_img_ids = set()
        ann_id = 0

        # mpii bbox scales are normalized with factor 200.
        pixel_std = 200.

        for idx, ann in enumerate(self.anns):
            center = np.array(ann['center'], dtype=np.float32)
            scale = np.array([ann['scale'], ann['scale']],
                             dtype=np.float32) * pixel_std

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15. / pixel_std * scale[1]

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1

            # unify shape with coco datasets
            center = center.reshape(1, -1)
            scale = scale.reshape(1, -1)
            bbox = bbox_cs2xyxy(center, scale)

            # load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            keypoints = np.array(ann['joints']).reshape(1, -1, 2)
            keypoints_visible = np.array(ann['joints_vis']).reshape(1, -1)

            instance_info = {
                'id': ann_id,
                'img_id': int(ann['image'].split('.')[0]),
                'img_path': osp.join(self.data_prefix['img'], ann['image']),
                'bbox_center': center,
                'bbox_scale': scale,
                'bbox': bbox,
                'bbox_score': np.ones(1, dtype=np.float32),
                'area': np.array((bbox[2]-bbox[0])*(bbox[3]-bbox[1])).reshape(1).astype(np.float64),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'iscrowd': 0,
                'category_id': np.array([1]),
            }

            if self.headbox_file:
                # calculate the diagonal length of head box as norm_factor
                headbox = headboxes_src[idx]
                head_size = np.linalg.norm(headbox[1] - headbox[0], axis=0)
                head_size *= SC_BIAS
                instance_info['head_size'] = head_size.reshape(1, -1)

            if instance_info['img_id'] not in used_img_ids:
                used_img_ids.add(instance_info['img_id'])
                image_list.append({
                    'img_id': instance_info['img_id'],
                    'img_path': instance_info['img_path'],
                })

            instance_list.append(instance_info)
            ann_id = ann_id + 1

        return instance_list, image_list


if __name__ == "__main__":
    # metainfo='/workspace/project/mmpose/configs/_base_/datasets/beta_mediapipe33.py',
    aa = MpiiDataset(
        dir='/workspace/dataset/human_keypoints/mpii', 
        ann_file='annotations/mpii_trainval.json',
        data_prefix=dict(img='images/')
    )
    num = len(aa)
    for i in range(num):
        data = aa[i]
        print(data)