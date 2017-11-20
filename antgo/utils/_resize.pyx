import numpy as np
cimport numpy as np
cdef _uint8_resize(unsigned char* data, src_size, target_size):
    cdef unsigned int height = src_size[0]
    cdef unsigned int width = src_size[1]
    cdef unsigned int channels = src_size[2]

    cdef unsigned int target_height = target_size[0]
    cdef unsigned int target_width = target_size[1]
    cdef np.ndarray[np.uint8_t, ndim=1] target_image = np.zeros((target_height * target_width * channels), dtype=np.uint8)
    cdef unsigned char* target_data = <unsigned char*> target_image.data

    cdef float scale_x =  float(width) / float(target_width)
    cdef float scale_y = float(height) / float(target_height)

    cdef unsigned int y = 0
    cdef unsigned int x = 0
    cdef float fy = 0.0
    cdef int sy = 0
    cdef float fx = 0.0
    cdef int sx = 0

    cdef unsigned int c = 0
    cdef int src_width_size = width * channels
    cdef int target_width_size = target_width * channels

    for y in range(target_height):
        fy = <float>((y + 0.5) * scale_y - 0.5)
        sy = <int> fy
        fy -= sy

        if fy < 0:
            fy = 0

        if sy < 0:
            fy = 0
            sy = 0
        if sy >= height -1:
            fy = 1
            sy = height -2

        for x in range(target_width):
            fx = <float>((x + 0.5) * scale_x - 0.5)
            sx = <int> fx
            fx -= sx

            if fx < 0:
                fx = 0

            if sx < 0:
                fx = 0
                sx = 0

            if sx >= width - 1:
                fx = 1
                sx = width - 2

            for c in range(channels):
                target_data[y * target_width_size + x * channels + c] = <unsigned char>((1.0 - fx) * (1.0 - fy) * data[sy*src_width_size+ sx*channels + c] + \
                                                                         (1.0 - fx) * fy * data[(sy+1) * src_width_size + sx*channels + c] + \
                                                                         fx * (1.0 - fy) * data[sy*src_width_size + (sx+1)*channels + c] + \
                                                                         fx * fy * data[(sy+1)*src_width_size + (sx+1)*channels + c] + 0.5)
    return target_image

cdef _f32_resize(float* data, src_size, target_size):
    cdef unsigned int height = src_size[0]
    cdef unsigned int width = src_size[1]
    cdef unsigned int channels = src_size[2]

    cdef unsigned int target_height = target_size[0]
    cdef unsigned int target_width = target_size[1]
    cdef np.ndarray[np.float32_t, ndim=1] target_image = np.zeros((target_height * target_width * channels), dtype=np.float32)
    cdef float* target_data = <float*> target_image.data

    cdef float scale_x =  float(width) / float(target_width)
    cdef float scale_y = float(height) / float(target_height)

    cdef unsigned int y = 0
    cdef unsigned int x = 0
    cdef float fy = 0.0
    cdef int sy = 0
    cdef float fx = 0.0
    cdef int sx = 0

    cdef unsigned int c = 0
    cdef int src_width_size = width * channels
    cdef int target_width_size = target_width * channels

    for y in range(target_height):
        fy = <float>((y + 0.5) * scale_y - 0.5)
        sy = <int> fy
        fy -= sy

        if fy < 0:
            fy = 0

        if sy < 0:
            fy = 0
            sy = 0
        if sy >= height -1:
            fy = 1
            sy = height -2

        for x in range(target_width):
            fx = <float>((x + 0.5) * scale_x - 0.5)
            sx = <int> fx
            fx -= sx

            if fx < 0:
                fx = 0

            if sx < 0:
                fx = 0
                sx = 0

            if sx >= width - 1:
                fx = 1
                sx = width - 2

            for c in range(channels):
                target_data[y * target_width_size + x * channels + c] = (1.0 - fx) * (1.0 - fy) * data[sy*src_width_size+ sx*channels + c] + \
                                                                         (1.0 - fx) * fy * data[(sy+1) * src_width_size + sx*channels + c] + \
                                                                         fx * (1.0 - fy) * data[sy*src_width_size + (sx+1)*channels + c] + \
                                                                         fx * fy * data[(sy+1)*src_width_size + (sx+1)*channels + c]
    return target_image


cdef _d64_resize(double* data, src_size, target_size):
    cdef unsigned int height = src_size[0]
    cdef unsigned int width = src_size[1]
    cdef unsigned int channels = src_size[2]

    cdef unsigned int target_height = target_size[0]
    cdef unsigned int target_width = target_size[1]
    cdef np.ndarray[np.double_t, ndim=1] target_image = np.zeros((target_height * target_width * channels), dtype=np.double)
    cdef double* target_data = <double*> target_image.data

    cdef float scale_x =  float(width) / float(target_width)
    cdef float scale_y = float(height) / float(target_height)

    cdef unsigned int y = 0
    cdef unsigned int x = 0
    cdef float fy = 0.0
    cdef int sy = 0
    cdef float fx = 0.0
    cdef int sx = 0

    cdef unsigned int c = 0
    cdef unsigned int src_width_size = width * channels
    cdef unsigned int target_width_size = target_width * channels

    for y in range(target_height):
        fy = <float>((y + 0.5) * scale_y - 0.5)
        sy = <int> fy
        fy -= sy

        if fy < 0:
            fy = 0

        if sy < 0:
            fy = 0
            sy = 0
        if sy >= height -1:
            fy = 1
            sy = height -2

        for x in range(target_width):
            fx = <float>((x + 0.5) * scale_x - 0.5)
            sx = <int> fx
            fx -= sx

            if fx < 0:
                fx = 0

            if sx < 0:
                fx = 0
                sx = 0

            if sx >= width - 1:
                fx = 1
                sx = width - 2

            for c in range(channels):
                target_data[y * target_width_size + x * channels + c] = (1.0 - fx) * (1.0 - fy) * data[sy*src_width_size+ sx*channels + c] + \
                                                                         (1.0 - fx) * fy * data[(sy+1) * src_width_size + sx*channels + c] + \
                                                                         fx * (1.0 - fy) * data[sy*src_width_size + (sx+1)*channels + c] + \
                                                                         fx * fy * data[(sy+1)*src_width_size + (sx+1)*channels + c]
    return target_image


def resize(image, target):
    cdef unsigned int height = image.shape[0]
    cdef unsigned int width = image.shape[1]
    cdef unsigned int channels = 1 if len(image.shape) == 2 else image.shape[2]
    cdef is_ndim_2 = 1 if len(image.shape) == 2 else 0
    cdef np.ndarray[np.double_t, ndim=1] double_image
    cdef np.ndarray[np.float32_t, ndim=1] float_image
    cdef np.ndarray[np.uint8_t, ndim=1] uint8_image

    if image.dtype == np.float or image.dtype == np.double or image.dtype == np.float64:
        double_image = np.ascontiguousarray(image.ravel(), dtype=np.double)
        target_image = _d64_resize(<double*>double_image.data, [height, width, channels], target)
        if is_ndim_2 == 1:
            return target_image.reshape(target)
        else:
            return target_image.reshape((target[0], target[1], channels))
    elif image.dtype == np.float32:
        float_image = np.ascontiguousarray(image.ravel(), dtype=np.float32)
        target_image = _f32_resize(<float*>float_image.data, [height, width, channels], target)
        if is_ndim_2 == 1:
            return target_image.reshape(target)
        else:
            return target_image.reshape((target[0], target[1], channels))
    if image.dtype == np.uint8:
        uint8_image = np.ascontiguousarray(image.ravel(), dtype=np.uint8)
        target_image = _uint8_resize(<unsigned char*>uint8_image.data, [height, width, channels], target)
        if is_ndim_2 == 1:
            return target_image.reshape(target)
        else:
            return target_image.reshape((target[0], target[1], channels))
    else:
        raise TypeError
