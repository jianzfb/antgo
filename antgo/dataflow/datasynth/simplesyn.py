import os
import cv2
import numpy as np


def perspective_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    new_pt[:2] = new_pt[:2] / new_pt[-1]
    return new_pt[:2]


def simple_syn_sample(image_generator, obj_generator, min_scale=0.5, max_scale=0.8):
    for image, obj_info in zip(image_generator, obj_generator):
        # obj_info {'image': , 'points': , 'mask': }

        # step 1. 图像调整到适合尺寸
        random_scale = np.random.random() * (max_scale - min_scale) + min_scale
        image_h, image_w, _ = image.shape

        object_image = obj_info['image']
        obj_h, obj_w, _ = object_image.shape

        min_size = max(image_w, image_h) * random_scale
        obj_scale = min_size / max(obj_h, obj_w)

        if obj_w * obj_scale > image_w * random_scale or obj_h * obj_scale > image_h * random_scale:
            if obj_w * obj_scale > image_w * random_scale:
                obj_scale = image_w * random_scale / obj_w

            if obj_h * obj_scale > image_h * random_scale:
                obj_scale = image_h * random_scale / obj_h

        object_image = cv2.resize(object_image, dsize=(int(obj_w * obj_scale), int(obj_h * obj_scale)))
        obj_h, obj_w = object_image.shape[:2]

        if 'points' in obj_info:
           obj_info['points'] = obj_info['points'] * obj_scale

        object_paste_mask = np.ones((obj_h, obj_w)).astype(np.uint8)
        if object_image.shape[-1] == 4:
            object_paste_mask = object_image[:,:, 3] / 255
            object_image = object_image[:,:,:3]

        if 'mask' in obj_info:
            obj_info['mask'] = cv2.resize(obj_info['mask'], dsize=(obj_w, obj_h))

        # step 2. 随机透射变换
        # 生成随机变换矩阵 3x3
        p1 = np.array([[0,0], [obj_w-1, 0], [0, obj_h-1], [obj_w-1, obj_h-1]]).astype(np.float32)
        tgt_p1 = [np.random.randint(0, obj_w//4), np.random.randint(0, obj_h//4)]
        tgt_p2 = [np.random.randint(obj_w-obj_w//4, obj_w), np.random.randint(0, obj_h//4)]
        tgt_p3 = [np.random.randint(0, obj_w//4), np.random.randint(obj_h-obj_h//4, obj_h)]
        tgt_p4 = [np.random.randint(obj_w-obj_w//4, obj_w), np.random.randint(obj_h-obj_h//4, obj_h)]
        p2 = np.array([tgt_p1, tgt_p2, tgt_p3, tgt_p4]).astype(np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)
        object_image = cv2.warpPerspective(object_image, M, (obj_w, obj_h))
        object_paste_mask = cv2. warpPerspective(object_paste_mask, M, (obj_w, obj_h))
        if 'points' in obj_info:
            for point_i in range(obj_info['points'].shape[0]):
                obj_info['points'][point_i, :2] = perspective_transform(obj_info['points'][point_i, :2], M)

        if 'mask' in obj_info:
            obj_info['mask'] = \
                cv2.warpPerspective(
                    obj_info['mask'],
                    M, 
                    (obj_w, obj_h), 
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=obj_info['fill'] if 'fill' in obj_info else 255)

        # 合成
        paste_x = 0
        if image_w > obj_w:
            paste_x = np.random.randint(0, image_w - obj_w)
        paste_y = 0
        if image_h > obj_h:
            paste_y = np.random.randint(0, image_h - obj_h)

        print(f'image {image_w} {image_h}')
        print(f'obj {obj_w} {obj_h}')
        object_paste_mask_expand = np.expand_dims(object_paste_mask, -1)
        image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask_expand) + object_image * object_paste_mask_expand

        if 'points' in obj_info:
            obj_info['points'] = obj_info['points'] + np.float32([[paste_x, paste_y]])

        if 'mask' in obj_info:
            fill_value = 255
            if 'fill' in obj_info:
                fill_value = obj_info['fill']

            mask = np.ones((image_h, image_w), dtype=np.uint8) * fill_value
            if 'hard' in obj_info:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_info['mask']
            else:
                mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = mask[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] * (1-object_paste_mask) + obj_info['mask'] * object_paste_mask

            obj_info['mask'] = mask

        # for joint_i, (x,y) in enumerate(obj_info['points']):
        #     x, y = int(x), int(y)
        #     image = cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)

        # cv2.imwrite(f'./abcd.png', image)
        yield image, obj_info

