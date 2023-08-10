import os
import cv2
import numpy as np


def perspective_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    new_pt[:2] = new_pt[:2] / new_pt[-1]
    return new_pt[:2]


def simple_syn_sample(image_generator, obj_generator, disturb_generator=None, min_scale=0.5, max_scale=0.8):
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
            obj_info['mask'] = cv2.resize(obj_info['mask'], dsize=(obj_w, obj_h), interpolation=cv2.INTER_NEAREST)

        # step 2. 随机透射变换
        # 生成随机变换矩阵 3x3
        p1 = np.array([[0,0], [obj_w-1, 0], [0, obj_h-1], [obj_w-1, obj_h-1]]).astype(np.float32)
        tgt_p1 = [np.random.randint(0, obj_w//8), np.random.randint(0, obj_h//8)]
        tgt_p2 = [np.random.randint(obj_w-obj_w//8, obj_w), np.random.randint(0, obj_h//8)]
        tgt_p3 = [np.random.randint(0, obj_w//8), np.random.randint(obj_h-obj_h//8, obj_h)]
        tgt_p4 = [np.random.randint(obj_w-obj_w//8, obj_w), np.random.randint(obj_h-obj_h//8, obj_h)]
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
                    borderValue=obj_info['fill'] if 'fill' in obj_info else 255,
                    flags=cv2.INTER_NEAREST)

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

        if disturb_generator is not None:
            # 干扰物生成器
            # disturb_image : HxWx3
            # disturb_mask : HxW 0～1浮点数
            disturb_image, disturb_mask, disturb_region = next(disturb_generator)

            # 分析干扰物
            disturb_pos = np.where(disturb_mask > 0.5)
            disturb_min_y = np.min(disturb_pos[0])
            disturb_min_x = np.min(disturb_pos[1])
            disturb_max_y = np.max(disturb_pos[0])
            disturb_max_x = np.max(disturb_pos[1])
            disturb_mask_rgb = (disturb_image * np.expand_dims(disturb_mask, -1)).astype(np.uint8)
            disturb_h = disturb_max_y - disturb_min_y
            disturb_scale = float(disturb_h) / float(disturb_image.shape[0])

            image_h, image_w = image.shape[:2]
            scaled_person_h = disturb_scale * image_h
            disturb_scale = scaled_person_h/disturb_h
            
            scaled_image_h = disturb_image.shape[0] * disturb_scale
            scaled_image_w = disturb_image.shape[1] * disturb_scale

            scaled_disturb_image = cv2.resize(disturb_mask_rgb, (int(scaled_image_w), int(scaled_image_h)))
            scaled_disturb_mask = cv2.resize(disturb_mask, (int(scaled_image_w), int(scaled_image_h)), interpolation=cv2.INTER_NEAREST)

            scaled_min_y = int(disturb_scale*disturb_min_y)
            scaled_min_x = int(disturb_scale*disturb_min_x)
            scaled_max_y = int(disturb_scale*disturb_max_y)
            scaled_max_x = int(disturb_scale*disturb_max_x)
            scaled_disturb_image = scaled_disturb_image[scaled_min_y:scaled_max_y, scaled_min_x:scaled_max_x]
            scaled_disturb_h, scaled_disturb_w = scaled_disturb_image.shape[:2]
            scaled_disturb_mask = scaled_disturb_mask[scaled_min_y:scaled_max_y, scaled_min_x:scaled_max_x]
            scaled_disturb_mask = np.expand_dims(scaled_disturb_mask, -1)

            is_paste_ok = False
            if 'mask' in obj_info:
                # 放置于mask区域
                mask_pos = np.where(obj_info['mask'] > 0.5)
                if len(mask_pos) > 0 or len(mask_pos[0]) > 0:
                    mask_min_y = np.min(mask_pos[0])
                    mask_min_x = np.min(mask_pos[1])
                    mask_max_y = np.max(mask_pos[0])
                    mask_max_x = np.max(mask_pos[1])

                    if disturb_region == 'center':
                        # 中心区域
                        cx = int((mask_min_x+mask_max_x)/2.0)
                        cy = int((mask_min_y+mask_max_y)/2.0)
                    else:
                        # 随机区域
                        cx = mask_min_x + np.random.random() * (mask_max_x - mask_min_x)
                        cy = mask_min_y + np.random.random() * (mask_max_y - mask_min_y)
                        cx = int(cx)
                        cy = int(cy)

                    start_y = cy-scaled_disturb_h
                    end_y = cy
                    start_x = cx-scaled_disturb_w
                    end_x = cx
                    try:
                        image[start_y:end_y, start_x:end_x] = \
                            scaled_disturb_image[:(end_y-start_y),:(end_x-start_x)] * scaled_disturb_mask[:(end_y-start_y),:(end_x-start_x)] + \
                            image[start_y:end_y, start_x:end_x] * (1-scaled_disturb_mask[:(end_y-start_y),:(end_x-start_x)])

                        is_paste_ok = True
                    except:
                        pass

        # for joint_i, (x,y) in enumerate(obj_info['points']):
        #     x, y = int(x), int(y)
        #     image = cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)

        # cv2.imwrite(f'./abcd.png', image)
        yield image, obj_info

