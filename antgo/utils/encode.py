# encoding=utf-8
# @Time    : 17-5-24
# @File    : encode.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from PIL import Image
try:
    # python 2
    from StringIO import StringIO as BufferIO
except ImportError:
    # python 3
    from io import BytesIO as BufferIO


def png_encode(data, thumbnail=False):
  assert(len(data.shape) == 2 or len(data.shape) == 3)
  if data.dtype != np.uint8:
    # normalized to 255
    max_val = float(np.max(data[:]))
    min_val = float(np.min(data[:]))
    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    data = norm_data.astype(np.uint8)

  # 1.step convert to Image
  image = Image.fromarray(data)

  if thumbnail:
    # 2.step (option) thumbnail
    width, height = image.size
    small_size = width if width < height else height
    ratio = small_size / 64
    small_width = int(width / ratio)
    small_height = int(height / ratio)
    image.thumbnail((small_width, small_height), Image.ANTIALIAS)

  # 2.step retarget to bufferio
  temp = BufferIO()
  image.save(temp, qtables='web_low', format='png')
  content = temp.getvalue()
  temp.close()

  return content