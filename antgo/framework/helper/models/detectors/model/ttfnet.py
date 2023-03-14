from antgo.framework.helper.models.builder import DETECTORS
from ..single_stage import SingleStageDetector
import torch
from torch import nn


@DETECTORS.register_module()
class TTFNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head,
                                     train_cfg, test_cfg, pretrained, init_cfg)

    def simple_test(self, img, img_metas, rescale=True, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        bbox_results = {
            'box': [a for a, _ in results_list],
            'label': [b for _, b in results_list],
        }
        # {'box', 'label'}
        return bbox_results

    def onnx_export(self, img):
        feat = self.extract_feat(img)
        local_cls, local_reg = self.bbox_head(feat)
        return local_cls, local_reg

