import enum
from antgo.framework.helper.models.builder import DETECTORS
from ..single_stage import SingleStageDetector, to_image_list
import torch
from torch import nn
import torch.nn.functional as F


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

    def simple_test(self, image, image_meta, rescale=True, **kwargs):
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
        image_list, image_meta = to_image_list(image, image_meta)
        image = image_list.tensors
        
        feat = self.extract_feat(image)
        results_list = self.bbox_head.simple_test(
            feat, image_meta, rescale=rescale)

        bbox_results = {
            'box': [a for a, _ in results_list],
            'label': [b for _, b in results_list],
        }
        # {'box', 'label'}
        return bbox_results

    def onnx_export(self, image):
        feat = self.extract_feat(image)
        local_cls, local_reg = self.bbox_head(feat)
        
        if isinstance(local_cls, list):
            for level_i, level_local_cls in enumerate(local_cls):
                level_local_cls = torch.sigmoid(level_local_cls)
                level_local_cls = F.max_pool2d(level_local_cls, 3, stride=1, padding=(3 - 1) // 2)
                local_cls[level_i] = level_local_cls
            
            return local_cls, local_reg
        else:
            local_cls = torch.sigmoid(local_cls)
            local_cls = F.max_pool2d(local_cls, 3, stride=1, padding=(3 - 1) // 2)
            return local_cls, local_reg
