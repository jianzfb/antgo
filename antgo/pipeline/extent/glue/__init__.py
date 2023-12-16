from . import backend
from .common import register, register_cstruct, CUSTOM_OP_LIST
from . import common
import ctypes
common.backend = backend

class CFTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_float)), False, 0)

    _fields_ = [("dim_size", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_float)), ('is_assign_inner', ctypes.c_bool), ('timestamp', ctypes.c_double)]


class CITensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int32)), False, 0)

    _fields_ = [("dim_size", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_int32)),  ('is_assign_inner', ctypes.c_bool), ('timestamp', ctypes.c_double)]


class CUCTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte)), False, 0)

    _fields_ = [("dim_size", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_ubyte)),  ('is_assign_inner', ctypes.c_bool), ('timestamp', ctypes.c_double)]


class CDTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_double)), False, 0)

    _fields_ = [("dim_size", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_double)), ('is_assign_inner', ctypes.c_bool), ('timestamp', ctypes.c_double)]


class CBTensor(ctypes.Structure):
    def __init__(self, data):
        self.val = data
        super().__init__(ctypes.c_size_t(len(data.shape)), ctypes.cast((ctypes.c_size_t * len(data.shape))(*data.shape), ctypes.POINTER(ctypes.c_size_t)), ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_bool)), False, 0)

    _fields_ = [("dim_size", ctypes.c_size_t), ("dims", ctypes.POINTER(ctypes.c_size_t)), ("data", ctypes.POINTER(ctypes.c_bool)), ('is_assign_inner', ctypes.c_bool), ('timestamp', ctypes.c_double)]


common.register_cstruct('CDTensor', CDTensor)
common.register_cstruct('CFTensor', CFTensor)
common.register_cstruct('CITensor', CITensor)
common.register_cstruct('CUCTensor', CUCTensor)
common.register_cstruct('CBTensor', CBTensor)