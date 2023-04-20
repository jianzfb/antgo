from antgo.measures import *
from ..runner.builder import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import copy
import json
import numpy as np
import time


class COCOWarp(COCO):
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


@MEASURES.register_module()
class COCOCompatibleEval(object):
    def __init__(self, categories, without_background=True):
        self.categories = categories
        for c in self.categories:
            if 'supercategory' not in c:
                c['supercategory'] = 'default'
        
        self.without_background = without_background
    
    def keys(self):
        # 约束使用此评估方法，需要具体的关键字信息
        return {'pred': ['box', 'label'], 'gt': ['image_meta', 'bboxes', 'labels']}

    def __call__(self, preds, gts):
        # gts 格式 'info', 'licenses', 'images', 'annotations', 'categories’
        # 将GT 转换为COCO格式
        # image_metas, bboxes, labels
        images = []
        annotations = []
        bbox_id = 0
        for image_id, gt in enumerate(gts):
            image_file = gt['image_meta']['image_file'] if 'image_file' in gt['image_meta'] else ''
            bboxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in gt['bboxes'].tolist()]
            areas = [box[2]*box[3] for box in bboxes]
            category_ids = [l for l in gt['labels'].tolist()]

            for _, (bbox, area, category_id) in enumerate(zip(bboxes, areas, category_ids)):
                if self.without_background:
                    category_id += 1

                annotations.append({
                    "segmentation": [],
                    "iscrowd": 0,
                    "image_id": image_id+1,
                    "bbox":bbox,
                    "area": area,
                    "category_id": category_id,
                    "id": bbox_id+1,
                    'ignore': 0
                })
                bbox_id += 1

            images.append({
                'height': gt['image_meta']['image_shape'][0],
                'width': gt['image_meta']['image_shape'][1],
                'id': image_id+1,
                'file_name': image_file
            })

        gt_coco = COCOWarp({
            'images': images,
            'categories': self.categories,
            'annotations': annotations
        })

        # 将预测转换为COCO格式
        pred_annotations = []
        for image_id, pred in enumerate(preds):
            pred_bboxes = pred['box'][:,:4]
            pred_probs = pred['box'][:,4]
            pred_labels = pred['label']

            for _, (pred_bbox, pred_prob, pred_label) in enumerate(zip(pred_bboxes, pred_probs, pred_labels)):
                x1, y1, x2, y2 = pred_bbox
                score = (float)(pred_prob)
                label = (int)(pred_label)
                if self.without_background:
                    label += 1

                w, h = x2 - x1, y2 - y1
                pred_dict = {
                    "image_id": image_id+1,
                    "bbox": [ x1, y1, w, h ],
                    "category_id": (int)(label),
                    "score": score
                }
                pred_annotations.append(pred_dict)

        pred_coco = gt_coco.loadRes(pred_annotations)

        coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
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
