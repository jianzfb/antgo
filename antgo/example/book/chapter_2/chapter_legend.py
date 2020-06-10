# -*- coding: UTF-8 -*-
# @Time    : 2019-02-08 09:17
# @File    : chapter_legend.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2

# joint bilateral filter legend
'''
input_img = np.zeros((50, 50))
input_img[:, 25:] = 1
input_img = input_img + (np.random.random((50, 50)) * 2 - 1)
plt.figure(1)
plt.imshow(input_img, cmap='gray')
plt.show()

guide_img = np.zeros((50, 50))
guide_img[:, 25:] = 1
plt.figure(2)
plt.imshow(guide_img, cmap='gray')
plt.show()

gaussian_kernel = np.zeros((20, 20))
for i in range(20):
  x_i = i - 10
  for j in range(20):
    x_j = j - 10
    gaussian_kernel[i, j] = np.exp(-((x_i*x_i) + x_j*x_j)/20.0)
gaussian_kernel = (gaussian_kernel - np.min(gaussian_kernel)) / (np.max(gaussian_kernel) - np.min(gaussian_kernel))

plt.imshow(gaussian_kernel, cmap='gray')
plt.show()

range_kernel = np.zeros((20,20))
range_kernel[:,10:] = 1
joint_kernel = gaussian_kernel * range_kernel
plt.imshow(joint_kernel, cmap='gray')
plt.show()


after_img = np.zeros((50, 50))
after_img[:, 25:] = 1
after_img = after_img + (np.random.random((50, 50)) * 2 - 1) / 10.0
plt.figure(1)
plt.imshow(after_img, cmap='gray')
plt.show()
'''
# portrait_img = cv2.imread('/Users/jian/Downloads/baidu00009.png')
# portrait_label = cv2.imread('/Users/jian/Downloads/baidu00009_label.png')
# cv2.imwrite('/Users/jian/Downloads/baidu00009_label_inv.png', (255-portrait_label).astype(np.uint8))
#
# portrait_mask = np.expand_dims(portrait_label[:,:,0],-1) / 255.0
#
# portrait_frontimg = portrait_img * portrait_mask
# cv2.imwrite('/Users/jian/Downloads/baidu00009_front.png', portrait_frontimg.astype(np.uint8))
#
# portrait_combine = cv2.imread('/Users/jian/Downloads/baidu00009_back.jpg')
# portrait_combine = cv2.resize(portrait_combine, (600, 800))
# portrait_combine = portrait_combine * (1-portrait_mask) + portrait_img * portrait_mask
# cv2.imwrite('/Users/jian/Downloads/baidu00009_back.png', portrait_combine.astype(np.uint8))

# ss = cv2.imread('/Users/jian/Downloads/00009.png')
# cv2.imshow('aa',(ss*255).astype(np.uint8))
# cv2.waitKey()

# ss = np.zeros((16,16))
# ss[6:10,5:10] = 1
# ss[0:2,0:2] = 1
# ss[14:16,14:16] = 1
# ss[0:2,14:16] = 1
# ss[14:16,0:2] = 1
# cv2.imwrite('/Users/jian/Downloads/ss.png', (ss*255).astype(np.uint8))
#
# ss_512 = cv2.resize(ss, (512,512))
# cv2.imwrite('/Users/jian/Downloads/ss_512.png', (ss_512*255).astype(np.uint8))
#


from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math


fig = plt.figure()

ax = fig.gca(projection='3d')
ax.view_init(elev=30., azim=20)

h = np.linspace(0,1,31)
s = np.linspace(0,1,9)
v = np.linspace(0,1,11)
h = h[1:-5]*2*np.pi

[H,V]=np.meshgrid(h,v)
cc=matplotlib.colors.hsv_to_rgb(np.dstack((H/2/np.pi,H-H+1,V)))
ax.plot_surface(V*np.cos(H),V*np.sin(H),V,facecolors=cc, shade=False)

[S,V]=np.meshgrid(s,v)
cc = matplotlib.colors.hsv_to_rgb(np.dstack((S-S+h[0]/2/np.pi,S,V)))
ax.plot_surface(S*V*np.cos(h[0]), S*V*np.sin(h[0]), V, rstride=1, cstride=1,facecolors=cc,shade=False,antialiased=False)

[S,V]=np.meshgrid(s,v)
cc = matplotlib.colors.hsv_to_rgb(np.dstack((S-S+h[-1]/2/np.pi,S,V)))
ax.plot_surface(S*V*np.cos(h[-1]), S*V*np.sin(h[-1]), V, rstride=1, cstride=1, facecolors=cc,shade=False,antialiased=False)


[H,S]=np.meshgrid(h,s)
cc=matplotlib.colors.hsv_to_rgb(np.dstack((H/2/np.pi, S, H-H+1)))
ax.plot_surface(S*np.cos(H), S*np.sin(H), H-H+1,rstride=1, cstride=1, facecolors=cc, shade=False,antialiased=False)

# shading flat;axis off equal
# view(60,15);camzoom(2);

[S,V]=np.meshgrid(s,v)
cc = matplotlib.colors.hsv_to_rgb(np.dstack((S-S+h[0]/2/np.pi,S,V)))
ax.plot_surface(S*V*np.cos(h[0]), S*V*np.sin(h[0]), V, rstride=1, cstride=1,facecolors=cc,shade=False,antialiased=False)

[S,V]=np.meshgrid(s,v)
cc = matplotlib.colors.hsv_to_rgb(np.dstack((S-S+h[-1]/2/np.pi,S,V)))
ax.plot_surface(S*V*np.cos(h[-1]), S*V*np.sin(h[-1]), V, rstride=1, cstride=1, facecolors=cc,shade=False,antialiased=False)

plt.axis('off')

plt.show()
