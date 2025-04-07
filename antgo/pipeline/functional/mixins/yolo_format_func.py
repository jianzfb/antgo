import json
import cv2
import os
import yaml

class YOLOFormatGen(object):
    def __init__(self, save_path, category_map, mode='detect', prefix="data"):
        self.save_path = save_path
        assert(mode in ['detect', 'pose', 'segment'])
        self.mode = mode
        self.class_ids = category_map
        self.inv_class_ids = {}
        for k,v in category_map.items():
            self.inv_class_ids[v] = k

        os.makedirs(os.path.join(self.save_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images', 'val'), exist_ok=True)
        self.image_folder = {
            'train': os.path.join(self.save_path, 'images', 'train'),
            'val': os.path.join(self.save_path, 'images', 'val')
        }

        os.makedirs(os.path.join(self.save_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'labels', 'val'), exist_ok=True)
        self.label_folder = {
            'train': os.path.join(self.save_path, 'labels', 'train'),
            'val': os.path.join(self.save_path, 'labels', 'val')
        }
        self.image_id = 0
        self.prefix = prefix

        with open(os.path.join(self.save_path, f'data.yaml'), "w", errors="ignore", encoding="utf-8") as f:
            data = {
                'path': './',
                'train': 'images/train',
                'val': 'images/val',
                'test': '',
                'names': self.inv_class_ids,
                'download': ''
            }
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def add(self, sample_info, stage='train'):
        if self.mode == 'detect':
            image = sample_info.image
            image_h, image_w = image.shape[:2]

            image_path = os.path.join(self.image_folder[stage], f'{self.prefix}-{self.image_id}.webp')
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])

            if len(sample_info.bboxes) > 0:
                label_path = os.path.join(self.label_folder[stage], f'{self.prefix}-{self.image_id}.txt')
                with open(label_path, 'w') as fp:
                    for box_i, box_info in enumerate(sample_info.bboxes):
                        x0,y0,x1,y1,c = 0, 0, 0, 0, None
                        if len(box_info) == 5:
                            x0,y0,x1,y1,c = box_info
                        else:
                            x0,y0,x1,y1 = box_info

                        if c is None:
                            c = int(sample_info.labels[box_i])

                        box=[float((x0+x1)/2.0/image_w),float((y0+y1)/2.0/image_h),float((x1-x0)/image_w),float((y1-y0)/image_h)]
                        fp.write(f'{int(c)} {box[0]} {box[1]} {box[2]} {box[3]}\n')
            else:
                label_path = os.path.join(self.label_folder[stage], f'{self.prefix}-{self.image_id}.txt')
                with open(label_path, 'w') as fp:
                    pass

            self.image_id += 1
            return
        elif self.mode == 'pose':
            # pose
            return
        else:
            # segment
            return

    def save(self):
        # do nothing
        pass