# encoding=utf-8
# @Time    : 17-5-24
# @File    : encode.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np


def png_encode(data):
  """ buf: must be bytes or a bytearray in Python3.x,
      a regular string in Python2.x.
  """
  import zlib, struct

  assert(len(data.shape) == 2 or len(data.shape) == 3)
  if data.dtype != np.uint8:
    # normalized to 255
    max_val = float(np.max(data[:]))
    min_val = float(np.min(data[:]))
    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    data = norm_data.astype(np.uint8)

  # data size
  width = data.shape[1]
  height = data.shape[0]

  # convert to buf
  aug_data = np.zeros((height,width,4),dtype=np.uint8)
  if len(data.shape) == 2:
    aug_data[:, :, 0:3] = np.tile(data,[1,1,3])
    aug_data[:,:,3] = 255
  else:
    assert(data.shape[2] == 3 or data.shape[2] == 4)
    aug_data[:,:,3] = 255
    aug_data[:,:,0:data.shape[2]] = data
  buf = bytearray(aug_data.flatten().tolist())

  # reverse the vertical line order and add null bytes at the start
  width_byte_4 = width * 4
  raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                      for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

  def png_pack(png_tag, data):
    chunk_head = png_tag + data
    return (struct.pack("!I", len(data)) +
            chunk_head +
            struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

  return b''.join([
    b'\x89PNG\r\n\x1a\n',
    png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
    png_pack(b'IDAT', zlib.compress(raw_data, 9)),
    png_pack(b'IEND', b'')])