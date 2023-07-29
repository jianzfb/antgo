# -*- coding: UTF-8 -*-
# @Time    : 2022/9/24 13:53
# @File    : holistic.py
# @Author  : jian<jian@mltalker.com>

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import os
import sys
import mediapipe


class Poselandmarker(object):
    def __init__(self, is_drawing=False, running_mode='IMAGE', num_poses=1, enable_segmentation=False, **kwargs):
        self.is_drawing = is_drawing
        self.mp_pose = mediapipe.solutions.pose.Pose(
                static_image_mode=True if running_mode=='IMAGE' else False,
                model_complexity=2,
                enable_segmentation=enable_segmentation,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        self.mp_drawing = mediapipe.solutions.drawing_utils
        self.mp_drawing_styles = mediapipe.solutions.drawing_styles

    def __call__(self, *args, **kwargs):
        image = args[0]
        image_h, image_w = image.shape[:2]
        results = self.mp_pose.process(image)

        pose_landmarks = np.zeros((len(results.pose_landmarks.landmark), 3), dtype=np.float32)
        for i in range(len(results.pose_landmarks.landmark)):
            pose_landmarks[i, 0] = results.pose_landmarks.landmark[i].x*image_w
            pose_landmarks[i, 1] = results.pose_landmarks.landmark[i].y*image_h
            pose_landmarks[i, 2] = results.pose_landmarks.landmark[i].visibility

        if self.is_drawing:
            # 绘图
            # Drawing pose
            image_drawing = image.copy()
            self.mp_drawing.draw_landmarks(
                image_drawing,
                results.pose_landmarks,
                mediapipe.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            return pose_landmarks, image_drawing
        
        return pose_landmarks