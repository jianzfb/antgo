import ctypes
import numpy as np
from .common import *

class NumPyTensor(MobulaTensor):
    @property
    def data_ptr(self):
        def p(e):
            return e.ctypes.data_as(ctypes.c_void_p)
        if not self.tensor.flags.c_contiguous:
            c = np.ascontiguousarray(self.tensor)
            return p(c), c
        return p(self.tensor)

    @property
    def ctype(self):
        return NPDTYPE2CTYPE(self.tensor.dtype)

    @property
    def dev_id(self):
        return None


# class OpGen(object):
#     def __init__(self, op, name):
#         self.op = op
#         self.name = name
#         self.cache = dict()

#     def __call__(self, *args, **kwargs):
#         if self.name not in self.cache:
#             # register operator
#             self.cache[self.name] = self.register()
#         try:
#             # forward and backward
#             kwargs.pop('__input_type__')
#             return self.cache[self.name](*args, **kwargs)
#         except KeyError:
#             # only forward
#             return self.cache[self.name]()(*args, **kwargs)

#     def register(self):
#         def forward(self, *args, **kwargs):
#             inputs, pars = get_in_data(op=self.op, *args, **kwargs)

#             self.in_data = inputs
#             dtype = self.in_data[0].dtype
#             self.req = ['write' for _ in self.in_data]
#             in_shape = get_in_shape(self.in_data)
#             out_shape = self.infer_shape(in_shape)[1]
#             self.out_data = [self.F.empty(s, dtype=dtype) for s in out_shape]
#             out = self._forward(*inputs)
#             if out is not None:
#                 if not isinstance(out, (list, tuple)):
#                     out = [out]
#                 for i, x in enumerate(out):
#                     self.assign(self.out_data[i], self.req[i], x)
#             if len(self.out_data) == 1:
#                 return self.out_data[0]
#             return self.out_data

#         def backward(self, out_grad=None, in_data=None, out_data=None, in_grad=None, req=None):

#             if in_data is not None:
#                 self.in_data = in_data
#             if out_data is not None:
#                 self.out_data = out_data

#             dtype = self.in_data[0].dtype

#             if in_grad is None:
#                 in_grad = [self.F.empty_like(d, dtype=dtype)
#                            for d in self.in_data]
#             else:
#                 if not isinstance(in_grad, (list, tuple)):
#                     in_grad = [in_grad]
#             self.in_grad = in_grad

#             if out_grad is None:
#                 out_grad = [self.F.ones_like(d, dtype=dtype)
#                             for d in self.out_data]
#             else:
#                 if not isinstance(out_grad, (list, tuple)):
#                     out_grad = [out_grad]
#             self.out_grad = out_grad

#             if req is None:
#                 self.req = ['write' for _ in self.in_data]
#             else:
#                 assert len(req) == len(self.in_data),\
#                     ValueError('len(req) should be %d' % len(self.in_data))
#                 self.req = req
#             out = self._backward(*out_grad)
#             if out is not None:
#                 if not isinstance(out, (list, tuple)):
#                     out = [out]
#                 num_inputs = len(get_varnames(self._forward))
#                 for i in range(num_inputs):
#                     self.assign(in_grad[i], self.req[i], out[i])
#             if len(in_grad) == 1:
#                 return in_grad[0]
#             return self.in_grad

#         np_op_dict = dict(
#             __call__=forward,
#             forward=forward,
#             backward=backward,
#             _forward=self.op.forward,
#             _backward=self.op.backward,
#             infer_shape=self.op.infer_shape,
#             assign=assign,
#             F=property(lambda self: np),
#             op=property(lambda dummy: self.op)
#         )
#         if hasattr(self.op, '__init__'):
#             np_op_dict['__init__'] = self.op.__init__
#         np_op_dict.update(INPUT_FUNCS)
#         np_op = type('_%s_NP_OP' % self.name,
#                      (self.op, object),
#                      np_op_dict)
#         return np_op


F = np
