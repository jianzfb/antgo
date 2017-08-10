# encoding=utf-8
# Time: 5/2/17
# File: objs_synth.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import random
import numpy as np
import math
from antgo.dataflow.dataset.dataset import *
from scipy.ndimage.interpolation import affine_transform

class ObjSynth(Dataset):
    def __init__(self, train_or_test, dir, ext_params=None):
        super(ObjSynth, self).__init__(train_or_test, dir, ext_params, 'ObjSynth')
        self.background_images = []
        self.objs_images = []
        self.epoch_size = int(getattr(self, 'epoch_size', 1000))
        self.max_deg = int(getattr(self, 'max_deg', 10))
        self.max_x_scale = float(getattr(self, 'max_x_scale', 1.0))
        self.max_y_scale = float(getattr(self, 'max_y_scale', 1.0))
        self.min_occupy = float(getattr(self, 'min_occupy', 0.1))
        self.max_occupy = float(getattr(self, 'max_occupy', 0.5))
        self.max_delta = float(getattr(self, 'max_delta', 20.0))
        self.sigma = float(getattr(self, 'sigma', 5.0))
        self.min_obj_size = int(getattr(self, 'min_obj_size', 25))
        self.is_random_rotation = int(getattr(self, 'is_random_rotation', 1))
        self.is_random_position = int(getattr(self, 'is_random_position', 1))
        self.is_random_scale = int(getattr(self, 'is_random_scale', 1))
        self.is_random_noise = int(getattr(self, 'is_random_noise', 1))
        self.is_random_brightness = int(getattr(self, 'is_random_brightness', 1))
        self.reset_state()

    def disable_random_rotation(self):
        self.is_random_rotation = False
    def enable_random_roration(self):
        self.is_random_rotation = True

    def disable_random_position(self):
        self.is_random_position = False
    def enable_random_position(self):
        self.is_random_position = True

    def diable_random_scale(self):
        self.is_random_scale = False
    def enable_random_scale(self):
        self.is_random_scale = True

    def disable_random_noise(self):
        self.is_random_noise = False
    def enable_random_noise(self):
        self.is_random_noise = True

    def disable_brightness(self):
        self.is_random_brightness = False
    def enable_brightness(self):
        self.is_random_brightness = True

    @property
    def rotation_max_deg(self):
        return self.max_deg
    @rotation_max_deg.setter
    def rotation_max_deg(self,val):
        self.max_deg = val

    @property
    def scale_x_max(self):
        return self.max_x_scale
    @scale_x_max.setter
    def scale_x_max(self,val):
        self.max_x_scale = val

    @property
    def scale_y_max(self):
        return self.max_y_scale
    @scale_y_max.setter
    def scale_y_max(self,val):
        self.max_y_scale = val

    @property
    def brightness_delta(self):
        return self.max_delta
    @brightness_delta.setter
    def brightness_delta(self,val):
        self.max_delta = val

    @property
    def gaussian_sigma(self):
        return self.sigma
    @gaussian_sigma.setter
    def gaussian_sigma(self,val):
        self.sigma = val

    @property
    def obj_min_size(self):
        return self.min_obj_size
    @obj_min_size.setter
    def obj_min_size(self,val):
        self.min_obj_size = val

    @property
    def obj_min_occupy(self):
        return self.min_occupy
    @obj_min_occupy.setter
    def obj_min_occupy(self,val):
        assert(val > 0.01)
        self.min_occupy = val

    @property
    def obj_max_occupy(self):
        return self.max_occupy
    @obj_max_occupy.setter
    def obj_max_occupy(self,val):
        assert(val < 1.0)
        self.max_occupy = val

    @property
    def sample_epoch_size(self):
        return self.epoch_size
    @sample_epoch_size.setter
    def sample_epoch_size(self,val):
        self.epoch_size = val

    def build(self, index=None, data=None):
        assert(os.path.exists(os.path.join(self.dir,self.train_or_test,'objs','annotation.txt')))
        assert(os.path.exists(os.path.join(self.dir,self.train_or_test,'backgrounds')))

        objs_folder = os.path.join(self.dir,self.train_or_test, 'objs')
        backgrounds_folder = os.path.join(self.dir,self.train_or_test, 'backgrounds')

        annotation_file = os.path.join(objs_folder,'annotation.txt')
        fp = open(annotation_file,'r')
        content = fp.readline()
        while content:
            file_name,file_mask_name,label = content.split(' ')
            self.objs_images.append((os.path.join(self.dir,self.train_or_test,'objs',file_name),
                                     os.path.join(self.dir, self.train_or_test, 'objs', file_mask_name),int(label)))

            content = fp.readline()
        fp.close()

        for background_file in os.listdir(backgrounds_folder):
            if 'png' in background_file or \
                            'jpg' in background_file or \
                            'bmp' in background_file:
                self.background_images.append(os.path.join(backgrounds_folder,background_file))

    @staticmethod
    def _affine_matrix(rotate_center,rotate_deg):
        theta = np.deg2rad(rotate_deg)
        transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        offset = np.array(rotate_center) - np.array(rotate_center).dot(transform)
        return transform, offset

    def data_pool(self):
        for index in range(self.epoch_size):
            # 1.step pick background and obj
            # pick background image randomly
            background_image_index = int(math.floor(random.random() * len(self.background_images)))
            background_img = self.load_image(self.background_images[background_image_index])
            background_width, background_height = background_img.shape[1::-1]

            # pick obj image randomly
            obj_img_index = int(math.floor(random.random() * len(self.objs_images)))

            # 2 step change obj image randomly
            obj_img_path = self.objs_images[obj_img_index][0]
            obj_mask_img_path = self.objs_images[obj_img_index][1]
            obj_label = self.objs_images[obj_img_index][2]
            obj_img = self.load_image(obj_img_path)
            obj_mask_img = self.load_image(obj_mask_img_path)[:,:,0]
            closely_bbox = np.where(obj_mask_img == 255)
            closely_bbox_y_min = np.min(closely_bbox[0])
            closely_bbox_y_max = np.max(closely_bbox[0])
            closely_bbox_x_min = np.min(closely_bbox[1])
            closely_bbox_x_max = np.max(closely_bbox[1])

            #
            obj_img = obj_img[closely_bbox_y_min:closely_bbox_y_max,closely_bbox_x_min:closely_bbox_x_max]
            obj_mask_img = obj_mask_img[closely_bbox_y_min:closely_bbox_y_max,closely_bbox_x_min:closely_bbox_x_max]

            obj_width, obj_height = obj_img.shape[1::-1]

            # 2.1 step GaussianNoise + Brightness
            obj_img = obj_img.astype(dtype=np.float32)
            brightness_delta = self.rng.uniform(-self.max_delta, self.max_delta)
            if self.is_random_brightness:
                obj_img += brightness_delta

            obj_img_noise = (self.rng.randn(obj_height, obj_width, obj_img.shape[2]) * 2 - 1) * self.sigma
            if self.is_random_noise:
                obj_img += obj_img_noise

            # 2.2 step random rotation angle
            if self.is_random_rotation:
                deg = self.rng.uniform(-self.max_deg, self.max_deg)
                rmat, offset = ObjSynth._affine_matrix([obj_width/2, obj_height/2], deg)
                obj_img_r,obj_img_g,obj_img_b = np.split(obj_img, 3, axis=2)
                obj_img_r = np.squeeze(obj_img_r, 2)
                obj_img_g = np.squeeze(obj_img_g, 2)
                obj_img_b = np.squeeze(obj_img_b, 2)
                obj_img_r = affine_transform(obj_img_r, rmat, offset=offset, output_shape=[obj_height, obj_width], cval= 0)
                obj_img_g = affine_transform(obj_img_g, rmat, offset=offset, output_shape=[obj_height, obj_width], cval= 0)
                obj_img_b = affine_transform(obj_img_b, rmat, offset=offset, output_shape=[obj_height, obj_width], cval= 0)
                obj_img = np.concatenate((np.expand_dims(obj_img_r,2),
                                          np.expand_dims(obj_img_g,2),
                                          np.expand_dims(obj_img_b,2)),axis=2)
                obj_mask_img = affine_transform(obj_mask_img,rmat,offset=offset,output_shape=[obj_height,obj_width],cval=0)
                obj_width, obj_height = obj_img.shape[1::-1]

            # 2.3 step resize
            if self.is_random_scale:
                max_x_scale = float(self.obj_max_occupy * background_width) / float(obj_width)
                min_x_scale = float(self.obj_min_occupy * background_width) / float(obj_width)
                if np.minimum(max_x_scale,self.max_x_scale) > min_x_scale:
                    max_x_scale = np.minimum(max_x_scale,self.max_x_scale)

                max_y_scale = float(self.obj_max_occupy * background_height) / float(obj_height)
                min_y_scale = float(self.obj_min_occupy * background_height) / float(obj_height)
                if np.minimum(max_y_scale,self.max_y_scale) > min_y_scale:
                    max_y_scale = np.minimum(max_y_scale,self.max_y_scale)

                x_scale = random.random() * (max_x_scale-min_x_scale) + min_x_scale
                y_scale = random.random() * (max_y_scale-min_y_scale) + min_y_scale
                to_width = int(math.floor(obj_width * x_scale))
                to_height = int(math.floor(obj_height * y_scale))

                obj_img = imresize(obj_img, (to_height,to_width))
                obj_mask_img = imresize(obj_mask_img, (to_height,to_width))
                obj_width = to_width
                obj_height = to_height

            # 2.4 step random position
            if self.is_random_position:
                x_pos = int(math.floor(random.random() * (background_width - obj_width)))
                y_pos = int(math.floor(random.random() * (background_height - obj_height)))
            else:
                x_pos = int(background_width - obj_width) / 2
                y_pos = int(background_height - obj_height) / 2

            # 3.step paste to background
            mask_obj = np.where(obj_mask_img[:, :] > 128)
            past_position_image = background_img[y_pos:y_pos + obj_height, x_pos:x_pos + obj_width]
            past_position_image[mask_obj[0], mask_obj[1], :] = obj_img[mask_obj[0], mask_obj[1], :]
            background_img[y_pos:y_pos + obj_height, x_pos:x_pos + obj_width] = past_position_image

            label = {'bbox': np.array([x_pos, y_pos, x_pos + obj_width, y_pos + obj_height]).reshape(1,4),
                     'category_id': np.array([obj_label]),
                     'category': [obj_label],
                     'flipped': False,
                     'area': np.array([obj_height * obj_width]),
                     'info': background_img.shape}

            yield [background_img, label]

    def split(self, train_validation_ratio=None, is_stratified_sampling=True):
        validation_set = ObjSynth('val', self.dir)
        return self, validation_set.build()