# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 20:39
# @File    : ocr.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config
from PIL import Image
import sys
import subprocess

try:
    import tesserocr
except ImportError:
    tesserocr = None

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.models.textdet.detectors import TextDetectorMixin
from mmocr.models.textrecog.recognizer import BaseRecognizer
from mmocr.utils import is_type_list
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm


class MMOCR:
    def __init__(self,
                 det_config='',
                 det_ckpt='',
                 recog_config='',
                 recog_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device=None,
                 **kwargs):

        self.device = device
        if self.device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.detect_model = init_detector(
            det_config, det_ckpt, device=self.device)
        self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = init_detector(
            recog_config, recog_ckpt, device=self.device)
        self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module


    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):
        args = locals().copy()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)

        # Input and output arguments processing
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model)
            pp_result = self.det_recog_pp(det_recog_result)
        else:
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                result = self.single_inference(model, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)
                pp_result = self.single_pp(result, model)

        return pp_result

    # Post processing function for end2end ocr
    def det_recog_pp(self, result):
        final_results = []
        args = self.args
        for arr, output, export, det_recog_result in zip(
                args.arrays, args.output, args.export, result):
            if output or args.imshow:
                if self.kie_model:
                    res_img = det_recog_show_result(arr, det_recog_result)
                else:
                    res_img = det_recog_show_result(
                        arr, det_recog_result, out_file=output)
                if args.imshow and not self.kie_model:
                    mmcv.imshow(res_img, 'inference results')
            if not args.details:
                simple_res = {}
                simple_res['filename'] = det_recog_result['filename']
                simple_res['text'] = [
                    x['text'] for x in det_recog_result['result']
                ]
                final_result = simple_res
            else:
                final_result = det_recog_result
            if export:
                mmcv.dump(final_result, export, indent=4)
            if args.print_result:
                print(final_result, end='\n\n')
            final_results.append(final_result)
        return final_results

    # Post processing function for separate det/recog inference
    def single_pp(self, result, model):
        for arr, output, export, res in zip(self.args.arrays, self.args.output,
                                            self.args.export, result):
            if export:
                mmcv.dump(res, export, indent=4)
            if output or self.args.imshow:
                if model == 'Tesseract_det':
                    res_img = TextDetectorMixin(show_score=False).show_result(
                        arr, res, out_file=output)
                elif model == 'Tesseract_recog':
                    res_img = BaseRecognizer.show_result(
                        arr, res, out_file=output)
                else:
                    res_img = model.show_result(arr, res, out_file=output)
                if self.args.imshow:
                    mmcv.imshow(res_img, 'inference results')
            if self.args.print_result:
                print(res, end='\n\n')
        return result

    def generate_kie_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            pred_score = node_pred_score[i]
            labels.append((pred_label, pred_score))
        return labels

    def visualize_kie_output(self,
                             model,
                             data,
                             result,
                             out_file=None,
                             show=False):
        """Visualizes KIE output."""
        img_tensor = data['img'].data
        img_meta = data['img_metas'].data
        gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
        if img_tensor.dtype == torch.uint8:
            # The img tensor is the raw input not being normalized
            # (For SDMGR non-visual)
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = tensor2imgs(
                img_tensor.unsqueeze(0), **img_meta.get('img_norm_cfg', {}))[0]
        h, w, _ = img_meta.get('img_shape', img.shape)
        img_show = img[:h, :w, :]
        model.show_result(
            img_show, result, gt_bboxes, show=show, out_file=out_file)

    # End2end ocr inference pipeline
    def det_recog_kie_inference(self, det_model, recog_model, kie_model=None):
        end2end_res = []
        # Find bounding boxes in the images (text detection)
        det_result = self.single_inference(det_model, self.args.arrays,
                                           self.args.batch_mode,
                                           self.args.det_batch_size)
        bboxes_list = [res['boundary_result'] for res in det_result]

        if kie_model:
            kie_dataset = KIEDataset(
                dict_file=kie_model.cfg.data.test.dict_file)

        # For each bounding box, the image is cropped and
        # sent to the recognition model either one by one
        # or all together depending on the batch_mode
        for filename, arr, bboxes, out_file in zip(self.args.filenames,
                                                   self.args.arrays,
                                                   bboxes_list,
                                                   self.args.output):
            img_e2e_res = {}
            img_e2e_res['filename'] = filename
            img_e2e_res['result'] = []
            box_imgs = []
            for bbox in bboxes:
                box_res = {}
                box_res['box'] = [round(x) for x in bbox[:-1]]
                box_res['box_score'] = float(bbox[-1])
                box = bbox[:8]
                if len(bbox) > 9:
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1:-1:2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1:-1:2])
                    box = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                box_img = crop_img(arr, box)
                if self.args.batch_mode:
                    box_imgs.append(box_img)
                else:
                    if recog_model == 'Tesseract_recog':
                        recog_result = self.single_inference(
                            recog_model, box_img, batch_mode=True)
                    else:
                        recog_result = model_inference(recog_model, box_img)
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, list):
                        text_score = sum(text_score) / max(1, len(text))
                    box_res['text'] = text
                    box_res['text_score'] = text_score
                img_e2e_res['result'].append(box_res)

            if self.args.batch_mode:
                recog_results = self.single_inference(
                    recog_model, box_imgs, True, self.args.recog_batch_size)
                for i, recog_result in enumerate(recog_results):
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, (list, tuple)):
                        text_score = sum(text_score) / max(1, len(text))
                    img_e2e_res['result'][i]['text'] = text
                    img_e2e_res['result'][i]['text_score'] = text_score

            if self.args.merge:
                img_e2e_res['result'] = stitch_boxes_into_lines(
                    img_e2e_res['result'], self.args.merge_xdist, 0.5)

            if kie_model:
                annotations = copy.deepcopy(img_e2e_res['result'])
                # Customized for kie_dataset, which
                # assumes that boxes are represented by only 4 points
                for i, ann in enumerate(annotations):
                    min_x = min(ann['box'][::2])
                    min_y = min(ann['box'][1::2])
                    max_x = max(ann['box'][::2])
                    max_y = max(ann['box'][1::2])
                    annotations[i]['box'] = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                ann_info = kie_dataset._parse_anno_info(annotations)
                ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                      ann_info['bboxes'])
                ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                     ann_info['bboxes'])
                kie_result, data = model_inference(
                    kie_model,
                    arr,
                    ann=ann_info,
                    return_data=True,
                    batch_mode=self.args.batch_mode)
                # visualize KIE results
                self.visualize_kie_output(
                    kie_model,
                    data,
                    kie_result,
                    out_file=out_file,
                    show=self.args.imshow)
                gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
                labels = self.generate_kie_labels(kie_result, gt_bboxes,
                                                  kie_model.class_list)
                for i in range(len(gt_bboxes)):
                    img_e2e_res['result'][i]['label'] = labels[i][0]
                    img_e2e_res['result'][i]['label_score'] = labels[i][1]

            end2end_res.append(img_e2e_res)
        return end2end_res

    # Separate det/recog inference pipeline
    def single_inference(self, model, arrays, batch_mode, batch_size=0):

        def inference(m, a, **kwargs):
            if model == 'Tesseract_det':
                return self.tesseract_det_inference(a)
            elif model == 'Tesseract_recog':
                return self.tesseract_recog_inference(a)
            else:
                return model_inference(m, a, **kwargs)

        result = []
        if batch_mode:
            if batch_size == 0:
                result = inference(model, arrays, batch_mode=True)
            else:
                n = batch_size
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                for chunk in arr_chunks:
                    result.extend(inference(model, chunk, batch_mode=True))
        else:
            for arr in arrays:
                result.append(inference(model, arr, batch_mode=False))
        return result

    # Arguments pre-processing function
    def _args_processing(self, args):
        # Check if the input is a list/tuple that
        # contains only np arrays or strings
        if isinstance(args.img, (list, tuple)):
            img_list = args.img
            if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
                raise AssertionError('Images must be strings or numpy arrays')

        # Create a list of the images
        if isinstance(args.img, str):
            img_path = Path(args.img)
            if img_path.is_dir():
                img_list = [str(x) for x in img_path.glob('*')]
            else:
                img_list = [str(img_path)]
        elif isinstance(args.img, np.ndarray):
            img_list = [args.img]

        # Read all image(s) in advance to reduce wasted time
        # re-reading the images for visualization output
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        num_res = len(img_list)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                args.output = [str(args.output)]
                if args.batch_mode:
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            args.output = [None] * num_res

        # If given an export argument, create a list of
        # result filenames for each image
        if args.export:
            export_path = Path(args.export)
            args.export = [
                str(export_path / f'out_{x}.{args.export_format}')
                for x in args.filenames
            ]
        else:
            args.export = [None] * num_res

        return args


class Ocr(object):
  def __init__(self, recog_config, det_config, recog_ckpt=None, det_ckpt=None, device='cpu'):
    # recog
    if not recog_config.endswith('.py'):
      recog_config = f'{recog_config}.py'
    recog_config_name = recog_config.split('.')[0]
    model_folder = os.path.join(os.environ['HOME'], '.antgo', 'models', 'mmocr')
    if not os.path.exists(model_folder):
      os.makedirs(model_folder)

    model_file = os.path.join(model_folder, recog_config)
    if not os.path.exists(model_file):
      subprocess.check_call([sys.executable, '-m', 'mim', 'download', 'mmocr', '--config', recog_config_name, '--dest', model_folder])

    recog_config_prefix_name = recog_config_name.split('_')[0]
    if recog_ckpt is None:
      for f in os.listdir(model_folder):
        if f.startswith(recog_config_prefix_name) and f.endswith('.pth'):
          recog_ckpt = f
          break
    assert(recog_ckpt is not None)

    # det
    if not det_config.endswith('.py'):
      det_config = f'{det_config}.py'
    det_config_name = det_config.split('.')[0]
    model_file = os.path.join(model_folder, det_config)
    if not os.path.exists(model_file):
      subprocess.check_call(
        [sys.executable, '-m', 'mim', 'download', 'mmocr', '--config', det_config_name, '--dest', model_folder])

    det_config_prefix_name = det_config_name.split('_')[0]
    if det_ckpt is None:
      for f in os.listdir(model_folder):
        if f.startswith(det_config_prefix_name) and f.endswith('.pth'):
          det_ckpt = f
          break
    assert (det_ckpt is not None)

    self.model = MMOCR(recog_ckpt=recog_ckpt, recog_config=recog_config,det_ckpt=det_ckpt,det_config=det_config, device=device)

  def __call__(self, *args, **kwargs):
    result = self.model.readtext(args[0], imshow=False)
    # label
    return result[0]
