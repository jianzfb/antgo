import copy
from abc import ABCMeta
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import numpy as np
import cv2
from numpy import random
from antgo.framework.helper.dataset.builder import PIPELINES
from antgo.framework.helper.dataset.pipelines.base import BaseTransform
from antgo.dataflow.structures.bbox.transforms import (bbox_clip_border, flip_bbox)
from antgo.dataflow.structures.keypoint.transforms import (flip_keypoints, keypoint_clip_border)


class MixImageTransform(BaseTransform, metaclass=ABCMeta):
    """Abstract base class for mixup-style image data augmentation.

    Args:
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 pre_transform: Optional[Sequence[str]] = None,
                 prob: float = 1.0):

        self.prob = prob

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        
        self.cache_info = []

    def transform(self, results: dict) -> dict:
        """Transform the input data dictionary using mixup-style augmentation.

        Args:
            results (dict): A dictionary containing input data.
        """
        if 'dataset' not in results and len(self.cache_info) < self.num_aux_image:
            self.cache_info.append(copy.deepcopy(results))
            if len(self.cache_info) < self.num_aux_image:
                results = self.centerpaste(results)
                return results

        results_cp = None
        if len(self.cache_info) > 0:
            if random.uniform(0, 1) < 0.5:
                results_cp = copy.deepcopy(results)

        if random.uniform(0, 1) < self.prob:
            dataset = results.pop('dataset', None)
            if dataset is not None:
                results['mixed_data_list'] = self._get_mixed_data_list(dataset)
            else:
                results['mixed_data_list'] = copy.deepcopy(self.cache_info)
            results = self.apply_mix(results)

            if 'mixed_data_list' in results:
                results.pop('mixed_data_list')
            if dataset is not None:
                results['dataset'] = dataset
        else:
            results = self.centerpaste(results)

        if len(self.cache_info) > 0:
            if results_cp is not None:
                index = np.random.choice(list(range(self.num_aux_image)))
                self.cache_info[index] = results_cp

        # temp_image = results['image'].copy()
        # for bbox in results['bboxes']:
        #     x0,y0,x1,y1 = bbox[:4]
        #     cv2.rectangle(temp_image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        
        # # for person_points,person_points_vis in zip(results['keypoints'], results['keypoints_visible']):
        # #     for point, point_vis in zip(person_points, person_points_vis):
        # #         if int(point_vis) == 1:
        # #             px,py = point[:2]
        # #             cv2.circle(temp_image, (int(px), int(py)), radius=2, color=(0,0,255), thickness=1)
        # cv2.imwrite('./a.png', temp_image)
        return results

    def _get_mixed_data_list(self, dataset):
        """Get a list of mixed data samples from the dataset.

        Args:
            dataset: The dataset from which to sample the mixed data.

        Returns:
            List[dict]: A list of dictionaries containing mixed data samples.
        """
        indexes = [
            random.randint(0, len(dataset)) for _ in range(self.num_aux_image)
        ]

        mixed_data_list = [
            copy.deepcopy(dataset[index]) for index in indexes
        ]

        if self.pre_transform is not None:
            for i, data in enumerate(mixed_data_list):
                data.update({'dataset': dataset})
                _results = self.pre_transform(data)
                _results.pop('dataset')
                mixed_data_list[i] = _results

        return mixed_data_list


@PIPELINES.register_module()
class Mosaic(MixImageTransform):
    """Mosaic augmentation. This transformation takes four input images and
    combines them into a single output image using the mosaic technique. The
    resulting image is composed of parts from each of the four sub-images. The
    mosaic transform steps are as follows:

    1. Choose the mosaic center as the intersection of the four images.
    2. Select the top-left image according to the index and randomly sample
        three more images from the custom dataset.
    3. If an image is larger than the mosaic patch, it will be cropped.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

    Required Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Modified Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        pad_val (int): Pad value. Defaults to 114.
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """

    num_aux_image = 3

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_range: Tuple[float, float] = (0.5, 1.5),
        pad_val: float = 114.0,
        pre_transform: Sequence[dict] = None,
        prob: float = 1.0,
    ):

        super().__init__(pre_transform=pre_transform, prob=prob)

        self.img_scale = img_scale
        self.center_range = center_range
        self.pad_val = pad_val

    def centerpaste(self, results):
        img_scale_w, img_scale_h = self.img_scale
        img_scale_w, img_scale_h = img_scale_w * 2, img_scale_h * 2
        canvas = np.zeros((img_scale_h, img_scale_h, 3), dtype=np.uint8)
        image = results['image']
        image_h, image_w = image.shape[:2]
        scale = min(img_scale_h, img_scale_w) / max(image_h, image_w)

        resized_h, resized_w = int(image_h * scale), int(image_w * scale)
        resized_img = cv2.resize(image, (resized_w, resized_h))
        offset_x = (img_scale_w - resized_w)//2
        offset_y = (img_scale_h - resized_h)//2
        canvas[offset_y:offset_y+resized_h, offset_x:offset_x+resized_w] = resized_img

        results['image'] = canvas
        results['image_meta'] = {
            'image_shape': canvas.shape
        }
        bboxes = results['bboxes'] * scale
        bboxes[:,::2] = bboxes[:,::2] + offset_x
        bboxes[:,1::2] = bboxes[:,1::2] + offset_y
        results['bboxes'] = bboxes

        if 'keypoints' in results:
            keypoints = results['keypoints'] * scale
            keypoints[:,:,0] = keypoints[:,:,0] + offset_x
            keypoints[:,:,1] = keypoints[:,:,1] + offset_y
            results['keypoints'] = keypoints
        elif 'joints2d' in results:
            keypoints = results['joints2d'] * scale
            keypoints[:,:,0] = keypoints[:,:,0] + offset_x
            keypoints[:,:,1] = keypoints[:,:,1] + offset_y
            results['joints2d'] = keypoints
        return results

    def apply_mix(self, results: dict) -> dict:
        """Apply mosaic augmentation to the input data."""

        assert 'mixed_data_list' in results
        mixed_data_list = results.pop('mixed_data_list')
        assert len(mixed_data_list) == self.num_aux_image

        img, annos = self._create_mosaic_image(results, mixed_data_list)
        bboxes = annos['bboxes']
        kpts = annos['keypoints'] if 'keypoints' in annos else annos['joints2d']
        kpts_vis = annos['keypoints_visible'] if 'keypoints_visible' in annos else annos['joints_vis']

        if len(bboxes) > 0:
            bboxes = bbox_clip_border(bboxes, (2 * self.img_scale[0], 2 * self.img_scale[1]))
        if len(kpts) > 0:
            kpts, kpts_vis = keypoint_clip_border(kpts, kpts_vis, (2 * self.img_scale[0], 2 * self.img_scale[1]))

        results['image'] = img
        results['image_meta'] = {
            'image_shape': img.shape
        }
        results['bboxes'] = bboxes
        if 'bboxes_score' in results:
            results['bboxes_score'] = annos['bboxes_score']

        if 'category_id' in results:
            results['category_id'] = annos['category_id']
        elif 'labels' in results:
            results['labels'] = annos['labels']

        if 'keypoints' in results:
            results['keypoints'] = kpts
            results['keypoints_visible'] = kpts_vis
        elif 'joints2d' in results:
            results['joints2d'] = kpts
            results['joints_vis'] = kpts_vis

        if 'area' in annos:
            results['area'] = annos['area']
        return results

    def _create_mosaic_image(self, results, mixed_data_list):
        """Create the mosaic image and corresponding annotations by combining
        four input images."""

        # init mosaic image
        img_scale_w, img_scale_h = self.img_scale
        mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2), 3),
                             self.pad_val,
                             dtype=results['image'].dtype)

        # calculate mosaic center
        center = (int(random.uniform(*self.center_range) * img_scale_w),
                  int(random.uniform(*self.center_range) * img_scale_h))

        annos = defaultdict(list)
        locs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for loc, data in zip(locs, (results, *mixed_data_list)):
            # process image
            img = data['image']
            h, w = img.shape[:2]
            scale_ratio = min(img_scale_h / h, img_scale_w / w)
            img = cv2.resize(img,
                                (int(w * scale_ratio), int(h * scale_ratio)))

            # paste
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center, img.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img[y1_c:y2_c, x1_c:x2_c]
            padw = x1_p - x1_c
            padh = y1_p - y1_c

            # merge annotations
            bboxes_category_id = 'category_id' if 'category_id' in data else 'labels'
            if 'bboxes' in data:
                bboxes = data['bboxes']

                # rescale & translate
                bboxes *= scale_ratio
                bboxes[..., ::2] += padw
                bboxes[..., 1::2] += padh

                annos['bboxes'].append(bboxes)
                if 'bboxes_score' in data:
                    annos['bboxes_score'].append(data['bboxes_score'])
                annos[bboxes_category_id].append(data[bboxes_category_id])

            keypoint_key = 'keypoints' if 'keypoints' in data else 'joints2d'
            keypoint_visble_key = 'keypoints_visible' if 'keypoints_visible' in data else 'joints_vis'
            if keypoint_key in data:
                kpts = data[keypoint_key]

                # rescale & translate
                kpts *= scale_ratio
                kpts[..., 0] += padw
                kpts[..., 1] += padh

                annos[keypoint_key].append(kpts)
                annos[keypoint_visble_key].append(data[keypoint_visble_key])

            if 'area' in data:
                annos['area'].append(data['area'] * scale_ratio**2)

        for key in annos:
            annos[key] = np.concatenate(annos[key])
        return mosaic_img, annos

    def _mosaic_combine(
        self, loc: str, center: Tuple[float, float], image_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Determine the overall coordinates of the mosaic image and the
        specific coordinates of the cropped sub-image."""

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')

        x1, y1, x2, y2 = 0, 0, 0, 0
        cx, cy = center
        w, h = image_shape

        if loc == 'top_left':
            x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
            crop_coord = w - (x2 - x1), h - (y2 - y1), w, h
        elif loc == 'top_right':
            x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w,
                                                     self.img_scale[0] * 2), cy
            crop_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        elif loc == 'bottom_left':
            x1, y1, x2, y2 = max(cx - w,
                                 0), cy, cx, min(self.img_scale[1] * 2, cy + h)
            crop_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        else:
            x1, y1, x2, y2 = cx, cy, min(cx + w, self.img_scale[0] *
                                         2), min(self.img_scale[1] * 2, cy + h)
            crop_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

        return (x1, y1, x2, y2), crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_range={self.center_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class YOLOXMixUp(MixImageTransform):
    """MixUp data augmentation for YOLOX. This transform combines two images
    through mixup to enhance the dataset's diversity.

    Mixup Transform Steps:

        1. A random image is chosen from the dataset and placed in the
            top-left corner of the target image (after padding and resizing).
        2. The target of the mixup transform is obtained by taking the
            weighted average of the mixup image and the original image.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    Required Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Modified Keys:

    - img
    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform (Optional[Sequence[str]]): A sequence of transform
            to be applied before mixup. Defaults to None.
        prob (float): Probability of applying the mixup transformation.
            Defaults to 1.0.
    """
    num_aux_image = 1

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 bbox_clip_border: bool = True,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0):
        assert isinstance(img_scale, tuple)
        super().__init__(pre_transform=pre_transform, prob=prob)
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def centerpaste(self, results):
        img_scale_w, img_scale_h = self.img_scale
        canvas = np.zeros((img_scale_h, img_scale_h, 3), dtype=np.uint8)
        image = results['image']
        image_h, image_w = image.shape[:2]
        scale = min(img_scale_h, img_scale_w) / max(image_h, image_w)

        resized_h, resized_w = int(image_h * scale), int(image_w * scale)
        resized_img = cv2.resize(image, (resized_w, resized_h))
        offset_x = (img_scale_w - resized_w)//2
        offset_y = (img_scale_h - resized_h)//2
        canvas[offset_y:offset_y+resized_h, offset_x:offset_x+resized_w] = resized_img

        results['image'] = canvas
        results['image_meta'] = {
            'image_shape': canvas.shape
        }
        bboxes = results['bboxes'] * scale
        bboxes[:,::2] = bboxes[:,::2] + offset_x
        bboxes[:,1::2] = bboxes[:,1::2] + offset_y
        results['bboxes'] = bboxes

        if 'keypoints' in results:
            keypoints = results['keypoints'] * scale
            keypoints[:,:,0] = keypoints[:,:,0] + offset_x
            keypoints[:,:,1] = keypoints[:,:,1] + offset_y
            results['keypoints'] = keypoints
        elif 'joints2d' in results:
            keypoints = results['joints2d'] * scale
            keypoints[:,:,0] = keypoints[:,:,0] + offset_x
            keypoints[:,:,1] = keypoints[:,:,1] + offset_y
            results['joints2d'] = keypoints
        return results

    def apply_mix(self, results: dict) -> dict:
        """YOLOX MixUp transform function."""

        assert 'mixed_data_list' in results
        mixed_data_list = results.pop('mixed_data_list')
        assert len(mixed_data_list) == self.num_aux_image

        if 'keypoints' in mixed_data_list[0]:
            if mixed_data_list[0]['keypoints'].shape[0] == 0:
                return results
        elif 'joints2d' in mixed_data_list[0]:
            if mixed_data_list[0]['joints2d'].shape[0] == 0:
                return results

        img, annos = self._create_mixup_image(results, mixed_data_list)
        bboxes = annos['bboxes']
        kpts = annos['keypoints'] if 'keypoints' in annos else annos['joints2d']
        kpts_vis = annos['keypoints_visible'] if 'keypoints_visible' in annos else annos['joints_vis']

        h, w = img.shape[:2]
        if len(bboxes) > 0:
            bboxes = bbox_clip_border(bboxes, (w, h))
        if len(kpts) > 0:
            kpts, kpts_vis = keypoint_clip_border(kpts, kpts_vis, (w, h))

        results['image'] = img.astype(np.uint8)
        results['image_meta'] = {
            'image_shape': img.shape
        }
        results['bboxes'] = bboxes
        if 'category_id' in results:
            results['category_id'] = annos['category_id']
        elif 'labels' in results:
            results['labels'] = annos['labels']

        if 'bboxes_score' in results:
            results['bboxes_score'] = annos['bboxes_score']

        if 'keypoints' in results:
            results['keypoints'] = kpts
            results['keypoints_visible'] = kpts_vis
        elif 'joints2d' in results:
            results['joints2d'] = kpts
            results['joints_vis'] = kpts_vis

        if 'area' in annos:
            results['area'] = annos['area']
        return results

    def _create_mixup_image(self, results, mixed_data_list):
        """Create the mixup image and corresponding annotations by combining
        two input images."""

        aux_results = mixed_data_list[0]
        aux_img = aux_results['image']

        # init mixup image
        out_img = np.ones((self.img_scale[1], self.img_scale[0], 3),
                          dtype=aux_img.dtype) * self.pad_val
        annos = defaultdict(list)

        # Calculate scale ratio and resize aux_img
        scale_ratio = min(self.img_scale[1] / aux_img.shape[0],
                          self.img_scale[0] / aux_img.shape[1])
        aux_img = cv2.resize(aux_img, (int(aux_img.shape[1] * scale_ratio),
                                          int(aux_img.shape[0] * scale_ratio)))

        # Set the resized aux_img in the top-left of out_img
        out_img[:aux_img.shape[0], :aux_img.shape[1]] = aux_img

        # random rescale
        jit_factor = random.uniform(*self.ratio_range)
        scale_ratio *= jit_factor
        out_img = cv2.resize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # random flip
        is_filp = random.uniform(0, 1) > self.flip_ratio
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # random crop
        ori_img = results['image']
        aux_h, aux_w = out_img.shape[:2]
        h, w = ori_img.shape[:2]
        padded_img = np.ones((max(aux_h, h), max(aux_w, w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:aux_h, :aux_w] = out_img

        dy = random.randint(0, max(0, padded_img.shape[0] - h) + 1)
        dx = random.randint(0, max(0, padded_img.shape[1] - w) + 1)
        padded_cropped_img = padded_img[dy:dy + h, dx:dx + w]

        # mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img

        # merge annotations
        # bboxes
        bboxes = aux_results['bboxes'].copy()
        bboxes *= scale_ratio
        bboxes = bbox_clip_border(bboxes, (aux_w, aux_h))
        if is_filp:
            bboxes = flip_bbox(bboxes, [aux_w, aux_h], 'xyxy')
        bboxes[..., ::2] -= dx
        bboxes[..., 1::2] -= dy
        annos['bboxes'] = [results['bboxes'], bboxes]
        if 'bboxes_score' in results:
            annos['bboxes_score'] = [
                results['bboxes_score'], aux_results['bboxes_score']
            ]

        if 'category_id' in results:
            annos['category_id'] = [
                results['category_id'], aux_results['category_id']
            ]
        elif 'labels' in results:
            annos['labels'] = [
                results['labels'], aux_results['labels']
            ]

        # keypoints
        if 'keypoints' in results:
            kpts = aux_results['keypoints'] * scale_ratio
            kpts, kpts_vis = keypoint_clip_border(kpts,
                                                aux_results['keypoints_visible'],
                                                (aux_w, aux_h))
            if is_filp:
                kpts, kpts_vis = flip_keypoints(kpts, kpts_vis, (aux_w, aux_h),
                                                aux_results['flip_indices'])
            kpts[..., 0] -= dx
            kpts[..., 1] -= dy
            annos['keypoints'] = [results['keypoints'], kpts]
            annos['keypoints_visible'] = [results['keypoints_visible'], kpts_vis]
        elif 'joints2d' in results:
            kpts = aux_results['joints2d'] * scale_ratio
            kpts, kpts_vis = keypoint_clip_border(kpts,
                                                aux_results['joints_vis'],
                                                (aux_w, aux_h))
            if is_filp:
                kpts, kpts_vis = flip_keypoints(kpts, kpts_vis, (aux_w, aux_h),
                                                aux_results['flip_indices'])
            kpts[..., 0] -= dx
            kpts[..., 1] -= dy
            annos['joints2d'] = [results['joints2d'], kpts]
            annos['joints_vis'] = [results['joints_vis'], kpts_vis]
        
        if 'area' in results:
            annos['area'] = [results['area'], aux_results['area'] * scale_ratio**2]

        for key in annos:
            annos[key] = np.concatenate(annos[key])
        return mixup_img, annos

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class FilterAnnotations(BaseTransform):
    """Eliminate undesirable annotations based on specific conditions.

    This class is designed to sift through annotations by examining multiple
    factors such as the size of the bounding box, the visibility of keypoints,
    and the overall area. Users can fine-tune the criteria to filter out
    instances that have excessively small bounding boxes, insufficient area,
    or an inadequate number of visible keypoints.

    Required Keys:

    - bbox (np.ndarray) (optional)
    - area (np.int64) (optional)
    - keypoints_visible (np.ndarray) (optional)

    Modified Keys:

    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground
            truth boxes. Default: (1., 1.)
        min_gt_area (int): Minimum foreground area of instances.
            Default: 1
        min_kpt_vis (int): Minimum number of visible keypoints. Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: False
        by_area (bool): Filter instances with area less than min_gt_area
            threshold. Default: False
        by_kpt (bool): Filter instances with keypoints_visible not meeting the
            min_kpt_vis threshold. Default: True
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_area: int = 1,
                 min_kpt_vis: int = 1,
                 by_box: bool = False,
                 by_area: bool = False,
                 by_kpt: bool = False,
                 keep_empty: bool = True) -> None:

        assert by_box or by_kpt or by_area
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_area = min_gt_area
        self.min_kpt_vis = min_kpt_vis
        self.by_box = by_box
        self.by_area = by_area
        self.by_kpt = by_kpt
        self.keep_empty = keep_empty

    def transform(self, results: dict) -> dict:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if self.by_box or self.by_area:
            if len(results['bboxes']) == 0:
                return results
        if self.by_kpt:
            if 'keypoints' in results:
                if len(results['keypoints']) == 0:
                    return results
            elif 'joints2d' in results:
                if len(results['joints2d']) == 0:
                    return results

        tests = []
        if self.by_box and 'bboxes' in results:
            bbox = results['bboxes']
            tests.append(
                ((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) &
                 (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
        if self.by_area and 'area' in results:
            area = results['area']
            tests.append(area >= self.min_gt_area)
        if self.by_kpt and (('keypoints' in results) or ('joints2d' in results)):
            if 'keypoints' in results:
                kpts_vis = results['keypoints_visible']
            else:
                kpts_vis = results['joints_vis']

            if kpts_vis.ndim == 3:
                kpts_vis = kpts_vis[..., 0]
            tests.append(kpts_vis.sum(axis=1) >= self.min_kpt_vis)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = ('bboxes', 'bboxes_score', 'category_id', 'keypoints', 'keypoints_visible', 'labels', 'joints2d', 'joints_vis', 'area')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        # temp_image = results['image'].copy()
        # for bbox, label in zip(results['bboxes'], results['labels']):
        #     x0,y0,x1,y1 = bbox[:4]
        #     if label == 0:
        #         cv2.rectangle(temp_image, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        #     else:
        #         cv2.rectangle(temp_image, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)

        # # for person_points,person_points_vis in zip(results['keypoints'], results['keypoints_visible']):
        # #     for point, point_vis in zip(person_points, person_points_vis):
        # #         if int(point_vis[0]) == 1:
        # #             px,py = point[:2]
        # #             cv2.circle(temp_image, (int(px), int(py)), radius=2, color=(0,0,255), thickness=1)
        # cv2.imwrite('./b.png', temp_image)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'min_gt_bbox_wh={self.min_gt_bbox_wh}, '
                f'min_gt_area={self.min_gt_area}, '
                f'min_kpt_vis={self.min_kpt_vis}, '
                f'by_box={self.by_box}, '
                f'by_area={self.by_area}, '
                f'by_kpt={self.by_kpt}, '
                f'keep_empty={self.keep_empty})')