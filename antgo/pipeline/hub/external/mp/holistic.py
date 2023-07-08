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

# Holistic(
#     static_image_mode=False, 
#     model_complexity=1, 
#     smooth_landmarks=True, 
#     min_detection_confidence=0.5, 
#     min_tracking_confidence=0.5
# )
class Holistic(object):
    def __init__(self, is_drawing=False, **kwargs):
        self.is_drawing = is_drawing
        self.holistic = mediapipe.solutions.holistic
        self.holistic_model = self.holistic.Holistic(**kwargs)
        self.drawing = mediapipe.solutions.drawing_utils

    def __call__(self, *args, **kwargs):
        image = args[0]
        image_h, image_w = image.shape[:2]
        results = self.holistic_model.process(image)

        image_drawing = None
        if self.is_drawing:
            image_drawing = image.copy()
            # Drawing the Facial Landmarks
            self.drawing.draw_landmarks(
                image_drawing,
                results.face_landmarks,
                self.holistic.FACEMESH_CONTOURS,
                self.drawing.DrawingSpec(
                    color=(255,0,255),
                    thickness=1,
                    circle_radius=1
                ),
                self.drawing.DrawingSpec(
                    color=(0,255,255),
                    thickness=1,
                    circle_radius=1
                )
            )

            # Drawing Right hand Land Marks
            self.drawing.draw_landmarks(
                image_drawing,
                results.right_hand_landmarks,
                self.holistic.HAND_CONNECTIONS
            )
        
            # Drawing Left hand Land Marks
            self.drawing.draw_landmarks(
                image_drawing,
                results.left_hand_landmarks,
                self.holistic.HAND_CONNECTIONS
            )

            # Drawing pose
            self.drawing.draw_landmarks(
                image_drawing,
                results.pose_landmarks,
                self.holistic.POSE_CONNECTIONS
            )

        face_landmarks = None
        if results.face_landmarks is not None:
            face_landmarks = np.zeros((len(results.face_landmarks.landmark), 3), dtype=np.float32)
            for i in range(len(results.face_landmarks.landmark)):
                face_landmarks[i, 0] = results.face_landmarks.landmark[i].x*image_w
                face_landmarks[i, 1] = results.face_landmarks.landmark[i].y*image_h
                face_landmarks[i, 2] = results.face_landmarks.landmark[i].visibility

        right_hand_landmarks = None
        if results.right_hand_landmarks is not None:
            right_hand_landmarks = np.zeros((len(results.right_hand_landmarks.landmark), 3), dtype=np.float32)
            for i in range(len(results.right_hand_landmarks.landmark)):
                right_hand_landmarks[i, 0] = results.right_hand_landmarks.landmark[i].x*image_w
                right_hand_landmarks[i, 1] = results.right_hand_landmarks.landmark[i].y*image_h
                right_hand_landmarks[i, 2] = results.right_hand_landmarks.landmark[i].visibility

        left_hand_landmarks = None
        if results.left_hand_landmarks is not None:
            left_hand_landmarks = np.zeros((len(results.left_hand_landmarks.landmark), 3), dtype=np.float32)
            for i in range(len(results.left_hand_landmarks.landmark)):
                left_hand_landmarks[i, 0] = results.left_hand_landmarks.landmark[i].x*image_w
                left_hand_landmarks[i, 1] = results.left_hand_landmarks.landmark[i].y*image_h
                left_hand_landmarks[i, 2] = results.left_hand_landmarks.landmark[i].visibility

        pose_landmarks = None
        if results.pose_landmarks is not None:
            pose_landmarks = np.zeros((len(results.pose_landmarks.landmark), 3), dtype=np.float32)
            for i in range(len(results.pose_landmarks.landmark)):
                pose_landmarks[i, 0] = results.pose_landmarks.landmark[i].x*image_w
                pose_landmarks[i, 1] = results.pose_landmarks.landmark[i].y*image_h
                pose_landmarks[i, 2] = results.pose_landmarks.landmark[i].visibility

        if self.is_drawing:
            return (face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks, image_drawing)
        else:
            return (face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks)