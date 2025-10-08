import json
import cv2
import os
import yaml
import numpy as np


class YOLOFormatGen(object):
    def __init__(self, save_path, category_map, mode='detect', prefix="data"):
        self.save_path = save_path
        assert(mode in ['detect', 'pose', 'segment', 'classify'])
        self.mode = mode
        self.class_ids = category_map
        self.inv_class_ids = {}
        if self.class_ids is not None:
            for k,v in self.class_ids.items():
                self.inv_class_ids[v] = k

        os.makedirs(os.path.join(self.save_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images', 'val'), exist_ok=True)
        self.image_folder = {
            'train': os.path.join(self.save_path, 'images', 'train'),
            'val': os.path.join(self.save_path, 'images', 'val')
        }

        if self.mode == 'segment':
            os.makedirs(os.path.join(self.save_path, 'masks', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, 'masks', 'val'), exist_ok=True)
            self.mask_folder = {
                'train': os.path.join(self.save_path, 'masks', 'train'),
                'val': os.path.join(self.save_path, 'masks', 'val')
            }

        os.makedirs(os.path.join(self.save_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'labels', 'val'), exist_ok=True)
        self.label_folder = {
            'train': os.path.join(self.save_path, 'labels', 'train'),
            'val': os.path.join(self.save_path, 'labels', 'val')
        }
        self.image_id = 0
        self.prefix = prefix
        self.is_update_config = False
        if self.mode == 'classify':
            # 对于分类任务，无需设置配置文件data.yaml
            return

        with open(os.path.join(self.save_path, f'data.yaml'), "w", errors="ignore", encoding="utf-8") as f:
            self.data = {
                'path': './',
                'train': 'images/train',
                'val': 'images/val',
                'test': '',
                'names': self.inv_class_ids,
                'download': ''
            }
            if self.mode == 'pose':
                self.data.update(
                    {
                        'kpt_shape': [],
                        'flip_idx': []
                    }
                )
            yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=True)

    def add(self, sample_info, stage='train'):
        image = sample_info.image
        if image is None:
            # invalid sample
            return
        image_h, image_w = image.shape[:2]

        sample_name = getattr(sample_info, 'img_path', None)
        if sample_name is not None:
            sample_name = sample_name.split('/')[-1]
        else:
            sample_name = f'{self.prefix}-{self.image_id}'

        print(sample_name)
        if self.mode == 'detect':
            # detect
            image_path = os.path.join(self.image_folder[stage], f'{sample_name}.webp')
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

            if len(sample_info.bboxes) > 0:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')
                with open(label_path, 'w') as fp:
                    for box_i, box_info in enumerate(sample_info.bboxes):
                        x0,y0,x1,y1,c = 0, 0, 0, 0, None
                        if len(box_info) == 5:
                            x0,y0,x1,y1,c = box_info
                        else:
                            x0,y0,x1,y1 = box_info

                        if c is None:
                            c = int(sample_info.labels[box_i])

                        x0 = max(x0, 0)
                        y0 = max(y0, 0)
                        x1 = min(x1, image_w)
                        y1 = min(y1, image_h)
                        box=[float((x0+x1)/2.0/image_w),float((y0+y1)/2.0/image_h),float((x1-x0)/image_w),float((y1-y0)/image_h)]
                        fp.write(f'{int(c)} {box[0]} {box[1]} {box[2]} {box[3]}\n')
            else:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')
                with open(label_path, 'w') as fp:
                    pass

            self.image_id += 1
            return
        elif self.mode == 'pose':
            # pose
            image_path = os.path.join(self.image_folder[stage], f'{sample_name}.webp')
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

            # 标注文件格式与 detect任务类似
            # cls box_cx box_cy box_w box_h x0 y0 ....
            if len(sample_info.bboxes) > 0:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')

                if not self.is_update_config:
                    # 根据数据信息，更新配置文件
                    with open(os.path.join(self.save_path, f'data.yaml'), "w", errors="ignore", encoding="utf-8") as f:
                        self.data.update(
                            {
                                'kpt_shape': [sample_info.joints2d.shape[1], 3],
                            }
                        )
                        yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=True)
                    self.is_update_config = True

                joints_vis = getattr(sample_info, 'joints_vis', None)
                if joints_vis is None:
                    joints_vis = np.ones((sample_info.joints2d.shape[0], sample_info.joints2d.shape[1], 1), dtype=np.int32)
                with open(label_path, 'w') as fp:
                    for box_i, (box_info, joints2d_info, joints2d_vis_info) in enumerate(zip(sample_info.bboxes, sample_info.joints2d, joints_vis)):
                        x0,y0,x1,y1,c = 0, 0, 0, 0, None
                        if len(box_info) == 5:
                            x0,y0,x1,y1,c = box_info
                        else:
                            x0,y0,x1,y1 = box_info

                        if c is None:
                            c = int(sample_info.labels[box_i])

                        x0 = max(x0, 0)
                        y0 = max(y0, 0)
                        x1 = min(x1, image_w)
                        y1 = min(y1, image_h)
                        box=[float((x0+x1)/2.0/image_w),float((y0+y1)/2.0/image_h),float((x1-x0)/image_w),float((y1-y0)/image_h)]
                        box_str = f'{int(c)} {box[0]} {box[1]} {box[2]} {box[3]}'

                        keypoint_str = ''
                        joints2d_info[:,0] = joints2d_info[:,0] / image_w
                        joints2d_info[:,1] = joints2d_info[:,1] / image_h

                        joints2d_info_ext = np.concatenate([joints2d_info, joints2d_vis_info], -1)
                        keypoint_str = ' '.join([str(float(v)) for v in joints2d_info_ext.flatten()])

                        anno_str = f'{box_str} {keypoint_str}\n'
                        fp.write(anno_str)
            else:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')
                with open(label_path, 'w') as fp:
                    pass

            self.image_id += 1
            return
        elif self.mode == 'classify':
            # classify
            if len(sample_info.bboxes) > 0:
                # 仅处理有目标
                for box_i, (box_info, label_info) in enumerate(zip(sample_info.bboxes, sample_info.labels)):
                    x0,y0,x1,y1,c = 0, 0, 0, 0, None
                    if len(box_info) == 5:
                        x0,y0,x1,y1,c = box_info
                    else:
                        x0,y0,x1,y1 = box_info

                    box_cx = (x0+x1)/2.0
                    box_cy = (y0+y1)/2.0
                    box_w = x1-x0
                    box_h = y1-y0
                    box_s = max(box_w, box_h)

                    x0 = int(max(box_cx - box_s/2, 0))
                    y0 = int(max(box_cy - box_s/2, 0))
                    x1 = int(min(box_cx + box_s/2, image_w))
                    y1 = int(min(box_cy + box_s/2, image_h))

                    box_image = image[y0:y1, x0:x1]
                    class_folder = os.path.join(self.image_folder[stage], str(label_info))
                    os.makedirs(class_folder, exist_ok=True)
                    image_path = os.path.join(class_folder, f'{sample_name}-{box_i}.webp')
                    cv2.imwrite(image_path, box_image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

            self.image_id += 1
            return
        else:
            # segment
            image_path = os.path.join(self.image_folder[stage], f'{sample_name}.webp')
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

            if len(sample_info.bboxes) > 0:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')
                with open(label_path, 'w') as fp:
                    for layout_i, (layout_label, layout_segment) in enumerate(zip(sample_info.labels, sample_info.segments)):
                        # segment_info -> polygon
                        label = layout_label + 1

                        mask = np.zeros((image_h, image_w), dtype=np.uint8)
                        mask[layout_segment == label] = 1
                        cv2.imwrite(os.path.join(self.mask_folder[stage], f'{sample_name}-{layout_label}.webp'), mask)

                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        polygons = []
                        for point_list in contours:
                            coords = []

                            for point in point_list:
                                coords.append(point[0][0] / image_w)
                                coords.append(point[0][1] / image_h)

                            coords_str = ' '.join([str(m) for m in coords])
                            anno_str = f'{layout_label} {coords_str}\n'
                            fp.write(anno_str)
            else:
                label_path = os.path.join(self.label_folder[stage], f'{sample_name}.txt')
                with open(label_path, 'w') as fp:
                    pass

            self.image_id += 1
            return

    def save(self):
        # do nothing
        pass