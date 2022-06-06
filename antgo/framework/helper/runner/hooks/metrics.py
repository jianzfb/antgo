from antgo.measures import *
from ..builder import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import copy
import json
import numpy as np
import time


class COCO_M(COCO):
    def __init__(self, gt_ann=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if gt_ann is not None:
            print('loading annotations into memory...')
            tic = time.time()
            if isinstance(gt_ann, str):
                with open(gt_ann, 'r') as f:
                    dataset = json.load(f)
                assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
                print('Done (t={:0.2f}s)'.format(time.time() - tic))
            else:
                dataset = gt_ann
            self.dataset = dataset
            self.createIndex()    

    def convert_from(self, pred_ann):
        '''
        predetect_result:为列表，每个列表中包含[x1, y1, x2, y2, score, label]
        img_name: 图片的名字
        '''
        json_data = []
        bbox_id = 100000
        for sample_i, sample_ann in enumerate(pred_ann):
            # sample_ann 包含bbox列表，和ann
            bboxes = sample_ann
            image_id = self.dataset['images'][sample_i]['id']

            for result in bboxes:
                x1, y1, x2, y2, score, label = result
                w, h = x2 - x1, y2 - y1
                # x1, y1 = x1, y1

                category_id = self.dataset['categories'][label]['id']
                detect_json = {
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [ x1, y1, w, h ],
                    "category_id": (int)(category_id),
                    "id": bbox_id,
                    "ignore": 0,
                    "segmentation": [],
                    "score": score
                }
                json_data.append(detect_json)

                bbox_id += 1

        anns = json_data
        res = COCO_M()
        res.dataset['images'] = [img for img in self.dataset['images']]

        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]

        res.dataset['annotations'] = anns
        res.createIndex()

        return res


@MEASURES.register_module()
class COCOBboxEval(object):
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, preds, gts):
        # gts 格式 'info', 'licenses', 'images', 'annotations', 'categories’
        coco_gt = COCO_M(gts)
        coco_dt = coco_gt.convert_from(preds)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        tags = [
            'AP@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
            'AP@[ IoU=0.50 | area= all | maxDets=100 ]',
            'AP@[ IoU=0.75 | area= all | maxDets=100 ]',
            'AP@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
            'AP@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
            'AP@[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
            'AR@[ IoU=0.50:0.95 | area= all | maxDets= 1 ]',
            'AR@[ IoU=0.50:0.95 | area= all | maxDets= 10 ]',
            'AR@[ IoU=0.50:0.95 | area= all | maxDets=100 ]',
            'AR@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
            'AR@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
            'AR@[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
        ]

        tag_and_value = {}
        for tag, value in zip(tags, coco_eval.stats.tolist()):
            tag_and_value[tag] = value

        return tag_and_value



# cc = COCOBboxEval()
# gt_ann = '/root/paddlejob/workspace/env_run/portrait/COCO/annotations/instances_val2017.json'
# with open(gt_ann, 'r') as f:
#     dataset = json.load(f)

# category_map_id = {}
# categories = dataset['categories']
# for ci, c in enumerate(categories):
#     category_map_id[c['id']] = ci

# image_map_id = {}
# images = dataset['images']
# for image_i, image in enumerate(images):
#     image_map_id[image['id']] = image_i

# annotations = dataset['annotations']
# det_result = [[] for _ in range(len(images))]
# for ann in annotations:
#     image_i = image_map_id[ann['image_id']]
#     x,y,w,h = ann['bbox']
#     label = category_map_id[ann['category_id']]
#     det_result[image_i].append([x,y,x+w,y+h,0.5,label])

# cc(det_result,dataset)
# print('aa')
