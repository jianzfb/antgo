from . import backend
from .common import register, register_cstruct, CUSTOM_OP_LIST
from . import common
import ctypes
common.backend = backend

class CFTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_float)))

    _fields_ = [("dim_num", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_float))]

class CITensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int)))

    _fields_ = [("dim_num", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_int))]

class CUCTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte)))

    _fields_ = [("dim_num", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_ubyte))]

common.register_cstruct('CFTensor', CFTensor)
common.register_cstruct('CITensor', CITensor)
common.register_cstruct('CUCTensor', CUCTensor)
