import numpy as np
cimport numpy as np

cdef extern from "lodepng.h":
  void lodepng_encode24(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h)

def encode_png(np.ndarray[np.uint8_t, ndim=3] image):
  # cdef np.ndarray[np.uint8_t, ndim=3] uint8_image
  # cdef unsigned char* uint8_image_data
  # cdef unsigned int height = image.shape[0]
  # cdef unsigned int width = image.shape[1]
  #
  # uint8_image = np.ascontiguousarray(image.ravel(), dtype=np.uint8)
  # uint8_image_data = <unsigned char*>uint8_image.data
  #
  #

  return None


def decode_png(image):
  return None