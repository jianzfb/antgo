import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from antgo.framework.helper.dataset.pipelines.base import BaseTransform
from antgo.framework.helper.dataset.pipelines.utils import avoid_cache_randomness, cache_randomness
from scipy.stats import truncnorm
from antgo.framework.helper.dataset.builder import PIPELINES
from antgo.dataflow.structures.bbox.transforms import bbox_xyxy2cs, flip_bbox
from antgo.dataflow.structures.keypoint.transforms import flip_keypoints
from antgo.framework.helper.dataset.pipelines.codecs.udp_heatmap import UDPHeatmap
from antgo.framework.helper.dataset.pipelines.codecs.yolox_pose_generate import YOLOXPoseAnnotationProcessor


try:
    import albumentations
except ImportError:
    albumentations = None

Number = Union[int, float]


@PIPELINES.register_module()
class GetBBoxCenterScale(BaseTransform):
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()

        self.padding = padding

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if 'bbox_center' in results and 'bbox_scale' in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('Use the existing "bbox_center" and "bbox_scale"'
                              '. The padding will still be applied.')
            results['bbox_scale'] = results['bbox_scale'] * self.padding

        else:
            bbox = results['bboxes']
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str


@PIPELINES.register_module()
class RandomHalfBody(BaseTransform):
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 9,
                 min_upper_keypoints: int = 2,
                 min_lower_keypoints: int = 3,
                 padding: float = 1.5,
                 prob: float = 0.3,
                 upper_prioritized_prob: float = 0.7) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]

        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        return center, scale

    @cache_randomness
    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (num_upper < self.min_upper_keypoints
                        and num_lower < self.min_lower_keypoints):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_valid_ids if prefer_upper else lower_valid_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'],
            upper_body_ids=results['upper_body_ids'],
            lower_body_ids=results['lower_body_ids'])

        bbox_center = []
        bbox_scale = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(min_total_keypoints={self.min_total_keypoints}, '
        repr_str += f'min_upper_keypoints={self.min_upper_keypoints}, '
        repr_str += f'min_lower_keypoints={self.min_lower_keypoints}, '
        repr_str += f'padding={self.padding}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'upper_prioritized_prob={self.upper_prioritized_prob})'
        return repr_str


# @PIPELINES.register_module()
# class RandomBBoxTransform(BaseTransform):
#     r"""Rnadomly shift, resize and rotate the bounding boxes.

#     Required Keys:

#         - bbox_center
#         - bbox_scale

#     Modified Keys:

#         - bbox_center
#         - bbox_scale

#     Added Keys:
#         - bbox_rotation

#     Args:
#         shift_factor (float): Randomly shift the bbox in range
#             :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
#             where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
#             Defaults to 0.16
#         shift_prob (float): Probability of applying random shift. Defaults to
#             0.3
#         scale_factor (Tuple[float, float]): Randomly resize the bbox in range
#             :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
#         scale_prob (float): Probability of applying random resizing. Defaults
#             to 1.0
#         rotate_factor (float): Randomly rotate the bbox in
#             :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
#             to 80.0
#         rotate_prob (float): Probability of applying random rotation. Defaults
#             to 0.6
#     """

#     def __init__(self,
#                  shift_factor: float = 0.16,
#                  shift_prob: float = 0.3,
#                  scale_factor: Tuple[float, float] = (0.5, 1.5),
#                  scale_prob: float = 1.0,
#                  rotate_factor: float = 80.0,
#                  rotate_prob: float = 0.6) -> None:
#         super().__init__()

#         self.shift_factor = shift_factor
#         self.shift_prob = shift_prob
#         self.scale_factor = scale_factor
#         self.scale_prob = scale_prob
#         self.rotate_factor = rotate_factor
#         self.rotate_prob = rotate_prob

#     @staticmethod
#     def _truncnorm(low: float = -1.,
#                    high: float = 1.,
#                    size: tuple = ()) -> np.ndarray:
#         """Sample from a truncated normal distribution."""
#         return truncnorm.rvs(low, high, size=size).astype(np.float32)

#     @cache_randomness
#     def _get_transform_params(self, num_bboxes: int) -> Tuple:
#         """Get random transform parameters.

#         Args:
#             num_bboxes (int): The number of bboxes

#         Returns:
#             tuple:
#             - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
#             - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
#             - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
#         """
#         random_v = self._truncnorm(size=(num_bboxes, 4))
#         offset_v = random_v[:, :2]
#         scale_v = random_v[:, 2:3]
#         rotate_v = random_v[:, 3]

#         # Get shift parameters
#         offset = offset_v * self.shift_factor
#         offset = np.where(
#             np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

#         # Get scaling parameters
#         scale_min, scale_max = self.scale_factor
#         mu = (scale_max + scale_min) * 0.5
#         sigma = (scale_max - scale_min) * 0.5
#         scale = scale_v * sigma + mu
#         scale = np.where(
#             np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

#         # Get rotation parameters
#         rotate = rotate_v * self.rotate_factor
#         rotate = np.where(
#             np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.)

#         return offset, scale, rotate

#     def transform(self, results: Dict) -> Optional[dict]:
#         """The transform function of :class:`RandomBboxTransform`.

#         See ``transform()`` method of :class:`BaseTransform` for details.

#         Args:
#             results (dict): The result dict

#         Returns:
#             dict: The result dict.
#         """
#         bbox_scale = results['bbox_scale']
#         num_bboxes = bbox_scale.shape[0]

#         offset, scale, rotate = self._get_transform_params(num_bboxes)

#         results['bbox_center'] = results['bbox_center'] + offset * bbox_scale
#         results['bbox_scale'] = results['bbox_scale'] * scale
#         results['bbox_rotation'] = rotate

#         return results

#     def __repr__(self) -> str:
#         """print the basic information of the transform.

#         Returns:
#             str: Formatted string.
#         """
#         repr_str = self.__class__.__name__
#         repr_str += f'(shift_prob={self.shift_prob}, '
#         repr_str += f'shift_factor={self.shift_factor}, '
#         repr_str += f'scale_prob={self.scale_prob}, '
#         repr_str += f'scale_factor={self.scale_factor}, '
#         repr_str += f'rotate_prob={self.rotate_prob}, '
#         repr_str += f'rotate_factor={self.rotate_factor})'
#         return repr_str


@PIPELINES.register_module()
@avoid_cache_randomness
class Albumentation(BaseTransform):
    """Albumentation augmentation (pixel-level transforms only).

    Adds custom pixel-level transformations from Albumentations library.
    Please visit `https://albumentations.ai/docs/`
    to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        transforms (List[dict]): A list of Albumentation transforms.
            An example of ``transforms`` is as followed:
            .. code-block:: python

                [
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                ]
        keymap (dict | None): key mapping from ``input key`` to
            ``albumentation-style key``.
            Defaults to None, which will use {'image': 'image'}.
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'image': 'image',
            }
        else:
            self.keymap_to_albu = keymap

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            albumentations.BasicTransform: The constructed transform object
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmengine.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            rank, _ = get_dist_info()
            if rank == 0 and not hasattr(
                    albumentations.augmentations.transforms, obj_type):
                warnings.warn(
                    f'{obj_type} is not pixel-level transformations. '
                    'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`Albumentation` to apply
        albumentations transforms.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): Result dict from the data pipeline.

        Return:
            dict: updated result dict.
        """
        # map result dict to albumentations format
        results_albu = {}
        for k, v in self.keymap_to_albu.items():
            assert k in results, \
                f'The `{k}` is required to perform albumentations transforms'
            results_albu[v] = results[k]

        # Apply albumentations transforms
        results_albu = self.aug(**results_albu)

        # map the albu results back to the original format
        for k, v in self.keymap_to_albu.items():
            results[k] = results_albu[v]

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class PhotometricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        """Generate the random flags for subsequent transforms.

        Returns:
            Sequence[Number]: a sequence of numbers that indicate whether to
                do the corresponding transforms.
        """
        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        contrast_mode = np.random.randint(2)
        # whether to apply brightness distortion
        brightness_flag = np.random.randint(2)
        # whether to apply contrast distortion
        contrast_flag = np.random.randint(2)
        # the mode to convert color from BGR to HSV
        hsv_mode = np.random.randint(4)
        # whether to apply channel swap
        swap_flag = np.random.randint(2)

        # the beta in `self._convert` to be added to image array
        # in brightness distortion
        brightness_beta = np.random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
        # the alpha in `self._convert` to be multiplied to image array
        # in contrast distortion
        contrast_alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        # the alpha in `self._convert` to be multiplied to image array
        # in saturation distortion to hsv-formatted img
        saturation_alpha = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        # delta of hue to add to image array in hue distortion
        hue_delta = np.random.randint(-self.hue_delta, self.hue_delta)
        # the random permutation of channel order
        swap_channel_order = np.random.permutation(3)

        return (contrast_mode, brightness_flag, contrast_flag, hsv_mode,
                swap_flag, brightness_beta, contrast_alpha, saturation_alpha,
                hue_delta, swap_channel_order)

    def _convert(self,
                 img: np.ndarray,
                 alpha: float = 1,
                 beta: float = 0) -> np.ndarray:
        """Multiple with alpha and add beta with clip.

        Args:
            img (np.ndarray): The image array.
            alpha (float): The random multiplier.
            beta (float): The random offset.

        Returns:
            np.ndarray: The updated image array.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PhotometricDistortion` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        assert 'image' in results, '`img` is not found in results'
        img = results['image']

        (contrast_mode, brightness_flag, contrast_flag, hsv_mode, swap_flag,
         brightness_beta, contrast_alpha, saturation_alpha, hue_delta,
         swap_channel_order) = self._random_flags()

        # random brightness distortion
        if brightness_flag:
            img = self._convert(img, beta=brightness_beta)

        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        if hsv_mode:
            # random saturation/hue distortion
            img = mmcv.bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                # apply saturation distortion to hsv-formatted img
                img[:, :, 1] = self._convert(
                    img[:, :, 1], alpha=saturation_alpha)
            if hsv_mode == 2 or hsv_mode == 3:
                # apply hue distortion to hsv-formatted img
                img[:, :, 0] = img[:, :, 0].astype(int) + hue_delta
            img = mmcv.hsv2bgr(img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_channel_order]

        results['image'] = img
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
    """

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = None
        if encoder['type'] == 'YOLOXPoseAnnotationProcessor':
            encoder.pop('type')
            self.encoder = YOLOXPoseAnnotationProcessor(**encoder)
        elif encoder['type'] == 'UDPHeatmap':
            encoder.pop('type')
            self.encoder = UDPHeatmap(**encoder)

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """
        keypionts_key = 'keypoints' if 'keypoints' in results else 'joints2d'
        keypoints_visible_key = 'keypoints_visible' if 'keypoints_visible' in results else 'joints_vis'
        keypoints_visible_weights_key = 'keypoints_visible_weights' if 'keypoints_visible' in results else 'joints_weight'

        keypoints = results[keypionts_key]
        keypoints_visible = results[keypoints_visible_key]
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = \
                keypoints_visible[..., 0], keypoints_visible[..., 1]
            results[keypoints_visible_key] = keypoints_visible
            results[keypoints_visible_weights_key] = keypoints_visible_weights

        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        auxiliary_encode_kwargs = {}
        if self.encoder.auxiliary_encode_keys is not None:
            for key in self.encoder.auxiliary_encode_keys:
                if key == 'category_id':
                    category_id_key = 'category_id' if 'category_id' in results else 'labels'
                    auxiliary_encode_kwargs['category_id'] = results[category_id_key]
                    continue
                auxiliary_encode_kwargs[key] = results[key]

        encoded = self.encoder.encode(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            **auxiliary_encode_kwargs)

        # temp_image = results['image'].copy()
        # for bbox in results['bboxes']:
        #     x0,y0,x1,y1 = bbox[:4]
        #     cv2.rectangle(temp_image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        
        # for person_points,person_points_vis in zip(results[keypionts_key], results[keypoints_visible_key]):
        #     for point, point_vis in zip(person_points, person_points_vis):
        #         if int(point_vis) == 1:
        #             px,py = point[:2]
        #             cv2.circle(temp_image, (int(px), int(py)), radius=2, color=(0,0,255), thickness=1)

        # cv2.imwrite('./d.png', temp_image)
        results.update(encoded)
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str


@PIPELINES.register_module()
class YOLOXHSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def transform(self, results: dict) -> dict:
        img = results['image']
        hsv_gains = self._get_hsv_gains()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results['image'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str
