import json
import cv2
import os
 
 
class COCOFormatGen(object):
    def __init__(self, save_path, category_map, mode='detect', prefix="data"):
        self.images = []
        self.categories = []
        self.annotations = []
        self.save_path = save_path
        assert(mode in ['detect', 'pose', 'segment'])
        self.mode = mode
        self.image_folder = os.path.join(self.save_path, 'image')
        os.makedirs(self.image_folder, exist_ok=True)
        self.anno_folder = os.path.join(self.save_path, 'anno')
        os.makedirs(self.anno_folder, exist_ok=True)

        self.class_ids = category_map
        self.inv_class_ids = {}
        for k,v in category_map.items():
            self.inv_class_ids[v] = k
        self.coco = {}

        for class_name, class_id in self.class_ids.items():
            self.categories.append(self.get_categories(class_name, class_id))
        self.coco['categories'] = self.categories

        self.image_id = 0
        self.anno_id = 0
        self.prefix = prefix
        self.stage = ''
 
    def add(self, sample_info, stage='train'):
        image = sample_info.image
        if image is None:
            # invalid sample
            return        
        image_h, image_w = image.shape[:2]
        self.stage = stage

        image_path = os.path.join(self.image_folder, f'{self.image_id}.webp')
        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
        self.images.append(self.get_images(f'{self.image_id}', image_h, image_w, self.image_id))
        for box_i, box_info in enumerate(sample_info.bboxes):
            x0,y0,x1,y1,c = 0, 0, 0, 0, None
            if len(box_info) == 5:
                x0,y0,x1,y1,c = box_info
            else:
                x0,y0,x1,y1 = box_info
            
            if c is None:
                c = int(sample_info.labels[box_i])
            box=[float(x0),float(y0),float(x1-x0),float(y1-y0)]
            class_name = self.inv_class_ids[int(c)]

            keypoints = [[1,1,1]]   # Nx3
            if 'joints2d' in sample_info.__dict__:
                keypoints = sample_info.joints2d[box_i].tolist()
            self.annotations.append(self.get_annotations(box, keypoints, self.image_id, self.anno_id, class_name))
            self.anno_id += 1
    
        self.image_id += 1

    def save(self):
        self.coco['images'] = self.images
        self.coco["annotations"] = self.annotations
    
        instances_train2017 = json.dumps(self.coco)
        f = open(os.path.join(self.anno_folder, f'{self.prefix}-{self.stage}.json'), 'w')
        f.write(instances_train2017)
        f.close()

    def get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = int(height)
        image['width'] = int(width)
        image["id"] = image_id
        image["file_name"] = filename+'.webp'
        return image
 
    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "Positive Cell"
        category['id'] = int(class_id)
        category['name'] = name
        return category
 
    def get_annotations(self, box, keypoints, image_id, ann_id, class_name):
        annotation = {}
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] = [[]]
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        annotation['category_id'] = self.class_ids[class_name]
        annotation['id'] = ann_id
        annotation['keypoints'] = keypoints
        return annotation
 