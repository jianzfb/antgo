# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 21:56
# @File    : computer_vision.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import cv2


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
    # from towhee.utils.cv2_utils import cv2
    cnt = limit

    def inner():
      nonlocal cnt
      cap = cv2.VideoCapture(device_id)
      while cnt != 0:
        retval, im = cap.read()
        if retval:
          yield im
          cnt -= 1

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
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
        yield frame

    return cls(inner())

  @classmethod
  def read_audio(cls, path):
    from towhee.utils.av_utils import av

    acontainer = av.open(path)

    audio_stream = acontainer.streams.audio[0]

    return cls(acontainer.decode(audio_stream))

  def to_video(self,
               output_path,
               codec=None,
               rate=None,
               width=None,
               height=None,
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成的参数
    out = cv2.VideoWriter(output_path, fourcc, rate, (width, height))  # 创建一个写入视频对象

    for array in self:
      out.write(array)

    out.release()
