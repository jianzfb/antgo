# -*- coding: UTF-8 -*-
# @Time    : 17-12-28
# @File    : image_tool.py
# @Author  : 
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def imshow(image, bboxes=None, keypoints=None):
  # Create figure and axes
  fig, ax = plt.subplots(1)
  
  # Display the image
  image = image.copy()
  if keypoints is not None:
    for k in range(keypoints.shape[0]):
      x, y = keypoints[k]
      image[int(y), int(x), :] = 255
      
  ax.imshow(image)
  
  if bboxes is not None:
    for b in range(bboxes.shape[0]):
      x0, y0, x1, y1 = bboxes[b]
      width = x1 - x0
      height = y1 - y0
      # Create a Rectangle patch
      rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
      # Add the patch to the Axes
      ax.add_patch(rect)

  plt.show()
  plt.waitforbuttonpress()