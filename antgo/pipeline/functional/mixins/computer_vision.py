# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 21:56
# @File    : computer_vision.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import cv2
import numpy as np
import os
import json
import uuid
from antgo.pipeline.functional.mixins.dag import register_dag
from antgo.pipeline.hparam import dynamic_dispatch
from antgo.pipeline.functional.entity import Entity
import traceback


class ComputerVisionMixin:
    """
    Mixin for computer vision problems.
    """
    def image_imshow(self, title='image'):  # pragma: no cover
        for im in self:
            cv2.imshow(title, im)
            cv2.waitKey(0)

    @classmethod
    def read_camera(cls, device_id=0, limit=-1):  # pragma: no cover
        """
        read images from a camera.
        """
        cnt = limit

        def inner():
            nonlocal cnt
            cap = cv2.VideoCapture(device_id)
            frame_count = 0
            while cnt != 0:
                retval, im = cap.read()
                if retval:
                    yield im, frame_count
                    cnt -= 1
                    frame_count += 1

        return cls(inner())

    # pylint: disable=redefined-builtin
    @classmethod
    def read_video(cls, path, format='rgb24'):
        """
        Load video as a datacollection.

        Args:
            path:
                The path to the target video.
            format:
                The format of the images loaded from video.
        """
        def inner():
            cap = cv2.VideoCapture(path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, frame_count
                frame_count += 1

        return cls(inner())

    @register_dag
    def video_decode(self, input_index, out_index):
        out_index = out_index + ('is_video_stop',)

        def inner():
            for video_info in self._iterable:
                video_file_path = getattr(video_info, input_index)
                if not os.path.exists(video_file_path):
                    print(f'path {video_file_path} not exist')
                    continue

                try:
                    frame_count = 0
                    cap = cv2.VideoCapture(video_file_path)
                    last_frame = None
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            frame_info = Entity(**{k:v for k,v in zip(out_index, (last_frame, frame_count, True))})
                            yield frame_info
                            break

                        last_frame = frame
                        frame_info = Entity(**{k:v for k,v in zip(out_index, (frame, frame_count, False))})
                        yield frame_info
                        frame_count += 1
                except:
                    traceback.print_exc()     

        return self._factory(inner())

    @register_dag
    def video_encode(self, input_index, out_index, output_folder, rate=30):
        if isinstance(input_index, tuple) or isinstance(input_index, list):
            input_index = input_index[0]
        
        if isinstance(out_index, tuple) or isinstance(out_index, list):
            out_index = out_index[0]

        def inner():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成的参数
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            while True:
                out = None
                video_path = ''
                try:
                    print(f'waiting video encoder start')
                    for info in self._iterable:
                        is_stop = getattr(info, 'is_video_stop')
                        if is_stop:
                            break
                        array = getattr(info, input_index)

                        h,w = array.shape[:2]
                        if out is None:
                            video_path = os.path.join(output_folder, f'{uuid.uuid4()}.mp4')
                            out = cv2.VideoWriter(video_path, fourcc, rate, (w, h))
                        out.write(array)

                    print(f'video {video_path} encoder finish')
                    out.release()
                    video_info = Entity(**{out_index:video_path})
                    yield video_info
                except:
                    traceback.print_exc()

        return self._factory(inner())

    def map_group(self, output_path, rows=10, cols=10):
        count = 0
        big_image = None
        init_width = -1
        init_height = -1
        shard = 0

        for x in self:
            row_i = count // cols
            col_i = (int)(count - row_i*cols)

            if len(x.shape) == 2:
                x = np.stack([x,x,x], -1)
            x = x[:,:,:3]
            height, width = x.shape[:2]
            big_width = (int)(cols * width)
            big_height = (int)(rows * height)
            if init_width < 0:
                init_width = width
            if init_height < 0:
                init_height = height

            if big_width != (int)(cols * init_width):
                big_width = (int)(cols * init_width)
            if big_height != (int)(rows * init_height):
                big_height = (int)(rows * init_height)

            if big_image is None:
                big_image = np.zeros((big_height, big_width, 3), dtype=np.uint8) 
            elif (row_i+1)*height > big_image.shape[0]:
                # 保存
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(os.path.join(output_path, f'{shard}.png'), big_image)
                shard += 1

                big_image = np.zeros((big_height, big_width, 3), dtype=np.uint8) 
                count = 0
                row_i = count // cols
                col_i = (int)(count - row_i*cols)

            # 填充当前图片
            big_image[row_i*height: row_i*height+x.shape[0], col_i*width:col_i*width+x.shape[1]] = x
            count += 1

    def to_video(self,
                output_path,
                width,
                height,
                rate=30,
                codec=None,
                format=None,
                template=None,
                audio_src=None):
        """
        Encode a video with audio if provided.

        Args:
            output_path:
                The path of the output video.
            codec:
                The codec to encode and decode the video.
            rate:
                The rate of the video.
            width:
                The width of the video.
            height:
                The height of the video.
            format:
                The format of the video frame image.
            template:
                The template video stream of the ouput video stream.
            audio_src:
                The audio to encode with the video.
        """
        assert(width is not None and height is not None)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成的参数
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')    
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        out = cv2.VideoWriter(output_path, fourcc, rate, (width, height))  # 创建一个写入视频对象

        for array in self:
            h,w = array.shape[:2]
            if h != height or w !=width:
                array = cv2.resize(array, (width, height))
            out.write(array)

        out.release()
        return output_path

    def to_json(self, path):
        total = []
        for data in self:
            assert(isinstance(data, dict))
            total.append(data)

        output_folder = os.path.dirname(path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(path, 'w') as fp:
            json.dump(total, fp)
