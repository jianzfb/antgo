from antgo.framework.helper.dataset.builder import PIPELINES
import numpy as np
import cv2
import os
import copy
import json


@PIPELINES.register_module()
class RandomNoiseAround(object):
    def __init__(self, prob=0.5, max_noise=20):
        super().__init__()
        self.prob = prob
        self.max_noise = max_noise
    
    def __call__(self, sample, context=None):
        if np.random.random() > self.prob:
            return sample
        
        gt_bbox = sample['gt_bbox']
        if gt_bbox.shape[0] == 0:
            return sample
        
        image = sample['image']
        image_h, image_w = image.shape[:2]
        
        image = image.astype(np.float32)
        for box in gt_bbox:
            x0,y0,x1,y1 = box.astype(np.int32)
            box_w = x1 - x0
            box_h = y1 - y0
            random_scale = np.random.random() * 0.5 + 0.5
            half_w = box_w * random_scale
            half_h = box_h * random_scale

            around_x0 = (int)(np.clip(x0 - half_w, 0, image_w))
            around_y0 = (int)(np.clip(y0 - half_h, 0, image_h))
            around_x1 = (int)(np.clip(x1 + half_w, 0, image_w))
            around_y1 = (int)(np.clip(y1 + half_h, 0, image_h))

            if around_x0 >= around_x1 or around_y0 >= around_y1:
                continue

            x0 = np.clip(x0, 0, image_w)
            y0 = np.clip(y0, 0, image_h)
            x1 = np.clip(x1, 0, image_w)
            y1 = np.clip(y1, 0, image_h)

            if x0 >= x1 or y0 >= y1:
                continue

            box_img = image[y0:y1,x0:x1].copy()
            image[around_y0:around_y1,around_x0:around_x1] += np.random.randint(0,self.max_noise, size=(around_y1-around_y0,around_x1-around_x0))
            image[y0:y1,x0:x1] = box_img

        image = image.astype(np.uint8)
        # cv2.imwrite("./temp/box.png", image)
        sample['image'] = image
        return sample



@PIPELINES.register_module()
class RandomPasteObject(object):
    def __init__(self, prob=0.5, obj_num=4) -> None:
        super().__init__()
        self.prob = prob
        self.obj_num = obj_num
    
    def __call__(self, sample, context=None):
        if np.random.random() < self.prob:
            # 随机贴干扰物
            gt_bbox = sample['gt_bbox']
            if gt_bbox.shape[0] == 0:
                return sample

            image = sample['image']
            for _ in range((int)(np.random.randint(1, self.obj_num+1))):
                random_i = np.random.randint(0,gt_bbox.shape[0], size=1)
                random_i = (int)(random_i)
                x0,y0,x1,y1 = gt_bbox[random_i]

                # 扩展划线框范围
                if np.random.random() < 0.5:
                    box_w = x1 - x0
                    box_h = y1 - y0

                    x0 = x0 - box_w
                    x1 = x1 + box_w
                    y0 = y0 - box_h
                    y1 = y1 + box_h

                # 随机选两条边
                random_edge_1 = np.random.randint(0,1, size=1)
                random_edge_1 = (int)(random_edge_1)
                random_edge_2 = np.random.randint(0,1,size=1)
                random_edge_2 = (int)(random_edge_2)

                noise_x0, noise_y0, noise_x1, noise_y1 = x0,y0,x1,y1
                if random_edge_1 == 0:
                    noise_x0 = np.random.random()*(x1-x0)+x0
                    noise_y0 = y0
                else:
                    noise_x0 = x0
                    noise_y0 = np.random.random()*(y1-y0)+y0
                
                if random_edge_2 == 0:
                    noise_x1 = np.random.random()*(x1-x0)+x0
                    noise_y1 = y1
                else:
                    noise_x1 = x1
                    noise_y1 = np.random.random()*(y1-y0)+y0
                
                noise_x0 = (int)(noise_x0)
                noise_y0 = (int)(noise_y0)
                noise_x1 = (int)(noise_x1)
                noise_y1 = (int)(noise_y1)
                image = cv2.line(image, (noise_x0, noise_y0),(noise_x1, noise_y1),(np.random.randint(0,255),0,0), np.random.randint(2, 5))
            sample['image'] = image

        return sample


@PIPELINES.register_module()
class RandomRotationInPico(object):
    """
    Rotate the image with bounding box
    """
    def __init__(self, degree=30, border_value=[0, 0, 0], label_border_value=0,inputs=None):
        super(RandomRotationInPico, self).__init__()
        self._degree = degree
        self._border_value = border_value
        self._label_border_value = label_border_value

    def __call__(self, sample, context=None):
        im = sample['image']
        height = sample['h']
        width = sample['w']
        cx, cy = width // 2, height // 2
        angle = np.random.randint(0, self._degree * 2) - self._degree
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        image_rotated = cv2.warpAffine(
                src=im,
                M=rot_mat,
                dsize=im.shape[1::-1],
                flags=cv2.INTER_AREA)

        if 'gt_bbox' in sample:
            gt_bbox = sample['gt_bbox']
            for i, bbox in enumerate(gt_bbox):
                x1, y1, x2, y2 = bbox
                coor = [[x1, x2, x1, x2], [y1, y1, y2, y2], [1, 1, 1, 1]]
                coor_new = np.matmul(rot_mat, coor)
                xmin = np.min(coor_new[0, :])
                ymin = np.min(coor_new[1, :])
                xmax = np.max(coor_new[0, :])
                ymax = np.max(coor_new[1, :])
                # region_scale = np.sqrt((xmax - xmin)*(ymax - ymin))
                # if region_scale > 50:
                #     margin = 1.8
                #     xmin = np.min(coor_new[0, :]) + np.abs(angle/margin)
                #     ymin = np.min(coor_new[1, :]) + np.abs(angle/margin)
                #     xmax = np.max(coor_new[0, :]) - np.abs(angle/margin)
                #     ymax = np.max(coor_new[1, :]) - np.abs(angle/margin)
                
                xmin = np.clip(xmin, 0, image_rotated.shape[1]-1)
                ymin = np.clip(ymin, 0, image_rotated.shape[0]-1)
                xmax = np.clip(xmax, 0, image_rotated.shape[1]-1)
                ymax = np.clip(ymax, 0, image_rotated.shape[0]-1)
                gt_bbox[i, 0] = xmin
                gt_bbox[i, 1] = ymin
                gt_bbox[i, 2] = xmax
                gt_bbox[i, 3] = ymax
            
            sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample and sample['gt_keypoint'].shape[0] > 0:
            gt_kpts = sample['gt_keypoint']
            for instance_i in range(gt_kpts.shape[0]):
                for i, kpt in enumerate(gt_kpts[instance_i]):
                    x1, y1, _ = kpt
                    coor = [[x1, x1, x1, x1], [y1, y1, y1, y1], [1, 1, 1, 1]]
                    coor_new = np.matmul(rot_mat, coor)
                    gt_kpts[instance_i, i, 0] = coor_new[0, 0]
                    gt_kpts[instance_i, i, 1] = coor_new[1, 0]

            sample['gt_keypoint'] = gt_kpts

        if 'gt_bodys' in sample['image_metas'] and sample['image_metas']['gt_bodys'].size != 0:
            for i,body_l in enumerate( sample['image_metas']['gt_bodys']):
                x1,y1=body_l
                coor = [[x1, x1, x1, x1], [y1, y1, y1, y1], [1, 1, 1, 1]]
                coor_new = np.matmul(rot_mat, coor)
                xmin = np.min(coor_new[0, :])
                ymin = np.min(coor_new[1, :])
                xmin = np.clip(xmin, 0, image_rotated.shape[1]-1)
                ymin = np.clip(ymin, 0, image_rotated.shape[0]-1)
                sample['image_metas']['gt_bodys'][i,0] = xmin
                sample['image_metas']['gt_bodys'][i,1] = ymin

        if 'semantic' in sample:
            label = cv2.warpAffine(
                sample['semantic'],
                M=rot_mat,
                dsize=im.shape[1::-1],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self._label_border_value)
            sample['semantic'] = label

        if 'image_metas' in sample and 'transform_matrix' in sample['image_metas']:
            # 记录加入的旋转矩阵
            t = np.eye(3)
            t[:2,:] = rot_mat
            sample['image_metas']['transform_matrix'] = np.matmul(t, sample['image_metas']['transform_matrix'])

        sample['h'] = image_rotated.shape[0]
        sample['w'] = image_rotated.shape[1]
        sample['image'] = image_rotated

        # # # 测试可视化
        # for bi in range(len(sample['gt_bbox'])):
        #     x0,y0,x1,y1 = sample['gt_bbox'][bi]
        #     cls_label = sample['gt_class'][bi]
        #     x0=(int)(x0)
        #     y0=(int)(y0)
        #     x1=(int)(x1)
        #     y1=(int)(y1)
        #     color_v = (255,0,0)
        #     if cls_label == 0:
        #         color_v = (0,0,255)
        #     cv2.rectangle(image_rotated, (x0,y0),(x1,y1), color_v, 4)
        # cv2.imwrite("./show.png", image_rotated)
        return sample



show_test_jian_i = 0
@PIPELINES.register_module()
class RandomFlipInPico(object):
    def __init__(self, 
                prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipInPico, self).__init__()
        self.prob = prob
        self.is_normalized = False

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:  
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) > self.prob:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        im = im[:, ::-1]
        if 'gt_bbox' in sample.keys() and sample['gt_bbox'].shape[0] > 0:
            gt_bbox = sample['gt_bbox']
            oldx1 = gt_bbox[:, 0].copy()
            oldx2 = gt_bbox[:, 2].copy()
            if self.is_normalized:
                gt_bbox[:, 0] = 1 - oldx2
                gt_bbox[:, 2] = 1 - oldx1
            else:
                gt_bbox[:, 0] = width - oldx2 - 1
                gt_bbox[:, 2] = width - oldx1 - 1

            sample['gt_bbox'] = gt_bbox
            selected_index = sample['gt_class'] <= 1
            sample['gt_class'][selected_index] = (1-sample['gt_class'][selected_index]).astype(np.int32)

        # if 'gt_person' in sample:
        #     person_p = sample['gt_person']
        #     if person_p[1] >= 0:
        #         x1, y1 = person_p
        #         x1 = width - x1

        #         person_p[0] = x1
        #         person_p[1] = y1
        #         sample['gt_person'] = person_p
                
        #         # im = im.copy()
        #         # cv2.circle(im, ((int)(x1), (int)(y1)), 5, (255,0,0))
        #         # cv2.imwrite('./temp/b.png', im[:,:,0].astype(np.uint8))
        
        #     sample['flipped'] = True
        #     sample['image'] = im

        sample['flipped'] = True
        sample['image'] = im
        if 'gt_bodys' in sample['image_metas'] and sample['image_metas']['gt_bodys'].size != 0:
            sample['image_metas']['gt_bodys'][:,0] = width - sample['image_metas']['gt_bodys'][:,0] -1

        if 'image_metas' in sample and 'transform_matrix' in sample['image_metas']:
            t = np.eye(3)
            t[0,0] = -1
            t[0,2] = width
            sample['image_metas']['transform_matrix'] = np.matmul(t, sample['image_metas']['transform_matrix'])
        
        if 'image_metas' in sample:
            sample['image_metas']['flipped'] = True

        # # debug
        # global show_test_jian_i
        # sample['image'] = sample['image'].copy()
        # debug_image = np.stack([sample['image'],sample['image'],sample['image']], -1)
        # for debug_i, debug_bbox in enumerate(sample['gt_bbox']):
        #     debug_x0, debug_y0, debug_x1, debug_y1 = debug_bbox
        #     debug_cx = (debug_x0+debug_x1)/2.0
        #     debug_cy = (debug_y0+debug_y1)/2.0
        #     debug_cls = sample['gt_class'][debug_i]
        #     bbox_belong_i = sample['image_metas']['gt_belongs'][debug_i]
        #     bbox_belong_i = (int)(bbox_belong_i)
        #     color = (255,0,0)
        #     if debug_cls == 0:
        #         color = (0, 0, 255)
        #     elif debug_cls == 1:
        #         color = (255,0,0)
        #     elif debug_cls == 2:
        #         color = (0,255,255)
        #     else:
        #         color = (0,255,0)
                    
        #     if sample['image_metas']['gt_bodys'].size != 0:
        #         if bbox_belong_i != -1:
        #             debug_body_x0, debug_body_y0 = sample['image_metas']['gt_bodys'][bbox_belong_i]
        #             cv2.circle(debug_image, ((int)(debug_body_x0),(int)(debug_body_y0)), 2, (255,0,0))
        #             cv2.line(debug_image, ((int)(debug_cx),(int)(debug_cy)),((int)(debug_body_x0),(int)(debug_body_y0)), color, 2)

        #     cv2.rectangle(debug_image, ((int)(debug_x0),(int)(debug_y0)), ((int)(debug_x1),(int)(debug_y1)), color, 2)
        
        # cv2.imwrite(f'./temp/show_{show_test_jian_i}.png', debug_image)
        # show_test_jian_i += 1
        return sample