# encoding=utf-8
# @Time    : 17-5-27
# @File    : image_with_tag.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def image_tagged_with_bboxes(image, bboxes, scores, labels, masks=[], keypoints=[], is_wait_press=True):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    # 0.step clear content
    plt.clf()
    # 1.step draw image
    plt.imshow(image)

    # 2.step draw bboxes
    for index in range(bboxes.shape[0]):
        bbox = bboxes[index, :]
        score = scores[index]
        label = labels[index]

        x0 = int(bbox[0])
        y0 = int(bbox[1])
        box_w = int(bbox[2] - bbox[0])
        box_h = int(bbox[3] - bbox[1])

        plt.axes().add_patch(patches.Rectangle((x0, y0), box_w, box_h, fill=None, color='C1', alpha=1))
        if type(label) != str:
            label = str(label)
        bbox_tag = "%s (%.2f)"%(label, score)
        plt.text(x0, y0, bbox_tag, fontsize=10, color='C1')

    if len(masks) > 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for obj_mask in masks:
            obj_mask_img = np.ones((obj_mask.shape[0], obj_mask.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                obj_mask_img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((obj_mask_img, obj_mask * 0.5)))

    plt.show(False)

    if is_wait_press:
        plt.waitforbuttonpress()