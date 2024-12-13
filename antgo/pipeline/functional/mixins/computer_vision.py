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
from antgo.utils.sample_gt import *
from antgo.tools.package import *
try:
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    # TODO，场景的路径规划需要自动导入
    from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger
except:
    print('no maniskill package import')
    pass


def _maniskill_episode_record(env, output_dir, prefix, save_video, episode_num):
    scene_name = env.spec.id
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name=prefix, 
        save_video=save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        save_on_reset=False
    )

    solve = globals()[f'solve{scene_name.split("-")[0]}']
    seed = 0
    episode_i = 0
    # 生成仿真环境操作轨迹
    while True and episode_i < episode_num:
        try:
            res = solve(env, seed=seed, debug=False, vis=False)
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1

        if res == -1:
            success = False
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()

        if not success:
            seed += 1
            env.flush_trajectory(save=False)
            env.flush_video(save=False)
            continue
        
        print(f"generate episode {episode_i}")
        env.flush_trajectory()
        env.flush_video()
        episode_i += 1

    env.close()


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
                width=None,
                height=None,
                rate=30,
                codec=None,
                format=None,
                template=None,
                audio_src=None, encode_code='mp4v'):
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
        # mp4v, VP80
        fourcc = cv2.VideoWriter_fourcc(*encode_code)  # 用于mp4格式的生成的参数
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')    
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        out = None
        if width is not None and height is not None:
            out = cv2.VideoWriter(output_path, fourcc, rate, (width, height))  # 创建一个写入视频对象

        frame_count = 0
        for data in self:
            if isinstance(data, Entity):
                data = list(data.__dict__.values())[0] 
            h,w = data.shape[:2]
            print(f'frame {frame_count}, shape ({h}, {w})')
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, rate, (w, h))

            if width is not None and height is not None:
                if h != height or w !=width:
                    dataarray = cv2.resize(data, (width, height))
            out.write(data)
            frame_count += 1

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

    def to_dataset(self, 
        folder, 
        prefix='train',
        is_tfrecord=False, 
        keymap=None):
        # 生成模型训练/测试/验证集
        # 主要用于创建感知模型（目标检测2D\3D、目标分割、深度估计、关键点2D\3D、图像分类）数据集
        if keymap is None:
            keymap={
                'image': 'image', 
                'segments': 'segments', 
                'joints2d': 'joints2d', 
                'joints3d': 'joints3d',
                'joints_vis': 'joints_vis',
                'labels': 'labels',
                'bboxes': 'bboxes', 
                'image_label': 'image_label',
                'clip_tag': 'clip_tag',
                'timestamp': 'timestamp',
                'bboxes3d_trans': 'bboxes3d_trans',
                'bboxes3d_rotation': 'bboxes3d_rotation',
                'bboxes3d_size': 'bboxes3d_size',
                'view_num': 'view_num',
                'view_id': 'view_id',
                "cam_param": 'cam_param'
            }

        # 对于objectdet（目标检测）任务，直接生成tfrecord或者json
        # 对于objectseg（目标分割）任务，直接生成tfrecord或者image/xxx.png, mask/xxx.png
        # 对于depth（深度估计）任务，直接生成tfrecord或者image/xxx.png, mask/xxx.png
        # 对于objlandmark（关键点估计）任务，直接生成tfrecord或者json
        # 对于imagecls（图像分类）任务，直接生成tfrecord或者json
        sgt = SampleGTTemplate()
        assert('image' in keymap)

        data_folder = os.path.join(folder, 'data')
        os.makedirs(data_folder, exist_ok=True)
        image_folder = os.path.join(data_folder, 'image')
        os.makedirs(image_folder, exist_ok=True)
        mask_folder = os.path.join(data_folder, 'mask')
        os.makedirs(mask_folder, exist_ok=True)

        tfrecord_folder = os.path.join(folder, 'tfrecord')
        if is_tfrecord:
            os.makedirs(tfrecord_folder, exist_ok=True)

        anno_info_list = []
        for index, info in enumerate(self):
            gt_info = sgt.get()

            image = info[keymap['image']]
            image_h, image_w = image.shape[:2]

            image_path = os.path.join(data_folder, 'image', f'{index}.png')
            cv2.imwrite(image_path, image)
            gt_info['image_file'] = f'data/image/{index}.png'
            gt_info['height'] = image_h
            gt_info['width'] = image_w

            mask_path = ''
            if 'segments' in keymap and keymap['segments'] in info:
                segments = info[keymap['segments']]
                mask_path = f'data/mask/{index}.png'
                cv2.imwrite(os.path.join(data_folder, 'mask', f'{index}.png'), segments)
            gt_info['semantic_file'] = mask_path

            joints2d = []
            if 'joints2d' in keymap and keymap['joints2d'] in info:
                joints2d = info[keymap['joints2d']]
            joints3d = []
            if 'joints3d' in keymap and keymap['joints3d'] in info:
                joints3d = info[keymap['joints3d']]
            joints_vis = []
            if 'joints_vis' in keymap and keymap['joints_vis'] in info:
                joints_vis = info[keymap['joints_vis']]

            gt_info['joints2d'] = joints2d
            gt_info['joints3d'] = joints3d
            gt_info['joints_vis'] = joints_vis

            bboxes = []
            labels = []
            if 'bboxes' in keymap and keymap['bboxes'] in info:
                bboxes = info[keymap['bboxes']]

            labels = []
            if 'labels' in keymap and keymap['labels'] in info:
                labels = info[keymap['labels']]
            gt_info['bboxes'] = bboxes
            gt_info['labels'] = labels

            image_label = -1
            if 'image_label' in keymap and keymap['image_label'] in info:
                image_label = info[keymap['image_label']]
            gt_info['image_label'] = image_label

            clip_tag = ''
            if 'clip_tag' in keymap and keymap['clip_tag'] in info:
                clip_tag = info[keymap['clip_tag']]
            timestamp = ""
            if 'timestamp' in keymap and keymap['timestamp'] in info:
                timestamp = info[keymap['timestamp']]
            gt_info['clip_tag'] = clip_tag
            gt_info['timestamp'] = timestamp

            view_num = 0
            if 'view_num' in keymap and keymap['view_num'] in info:
                view_num = info[keymap['view_num']]
            view_id = ""
            if 'view_id' in keymap and keymap['view_id'] in info:
                view_id = info[keymap['view_id']]
            cam_param = {}
            if 'cam_param' in keymap and keymap['cam_param'] in info:
                cam_param = info[keymap['cam_param']]
            gt_info['clip_tag'] = clip_tag
            gt_info['view_id'] = view_id
            gt_info['cam_param'] = cam_param

            bboxes3d_trans = []
            if 'bboxes3d_trans' in keymap and keymap['bboxes3d_trans'] in info:
                bboxes3d_trans = info[keymap['bboxes3d_trans']]
            bboxes3d_rotation = []
            if 'bboxes3d_rotation' in keymap and keymap['bboxes3d_rotation'] in info:
                bboxes3d_rotation = info[keymap['bboxes3d_rotation']]
            bboxes3d_size = []
            if 'bboxes3d_size' in keymap and keymap['bboxes3d_size'] in info:
                bboxes3d_size = info[keymap['bboxes3d_size']]
            gt_info['bboxes3d_trans'] = bboxes3d_trans
            gt_info['bboxes3d_rotation'] = bboxes3d_rotation
            gt_info['bboxes3d_size'] = bboxes3d_size
            anno_info_list.append(gt_info)

        with open(os.path.join(folder, f'{prefix}.json'), 'w') as fp:
            json.dump(anno_info_list, fp)

        if is_tfrecord:
            package_to_tfrecord(os.path.join(folder, f'{prefix}.json'), tfrecord_folder, prefix, size_in_shard=10000)

    # for robot env
    def solve(self, output_dir='./record', prefix='trajectory', save_video=True, episode_num=1):
        for env_info in self:
            env = list(env_info.__dict__.values())[0]
            if env['name'] == 'maniskill':
                _maniskill_episode_record(env['env'], output_dir, prefix, save_video, episode_num)
            env_info.env['status'] = 'done'

    # for robot env
    def drive(self):
        for env_ifo in self:
            pass
