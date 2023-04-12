# Copyright (c) OpenMMLab. All rights reserved.
import torch
from antgo.framework.helper.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .core.bbox import bbox2result
from antgo.framework.helper.runner import *


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        # if pretrained:
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     backbone.pretrained = pretrained
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops."""
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      image,
                      image_meta,                      
                      bboxes,
                      labels,
                      bboxes_ignore=None, **kwargs):
        """
        Args:
            image (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            image_meta (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            labels (list[Tensor]): Class indices corresponding to each box
            bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(image, image_meta)
        x = self.extract_feat(image)
        
        if bboxes is None or labels is None:
            return self.bbox_head(x)

        losses = self.bbox_head.forward_train(x, image_meta, bboxes,
                                              labels, bboxes_ignore)
        return losses
    
    def loss(self, *args, image_meta, bboxes, labels):
        losses = self.bbox_head.loss(*args, bboxes, labels, image_meta)
        return losses
    
    def simple_test(self, image, image_meta, rescale=True, **kwargs):
        """Test function without test-time augmentation.

        Args:
            image (torch.Tensor): Images with shape (N, C, H, W).
            image_meta (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(image)
        results_list = self.bbox_head.simple_test(
            feat, image_meta, rescale=rescale)

        bbox_results = {
            'box': torch.stack([a for a, _ in results_list], dim=0),
            'label': torch.stack([b for _, b in results_list], dim=0)
        }
        return bbox_results