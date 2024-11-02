from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            Z = Z.reshape((Z.size, ))
            self.axes = 0
        return (Z - Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)).exp().sum(axis=self.axes).log() + Z.max(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        max_z = Tensor(node.inputs[0].realize_cached_data().max(
                        axis=self.axes, 
                        keepdims=True), device=out_grad.device).broadcast_to(node.inputs[0].shape) #! 防止数值溢出
        exp_z = exp(node.inputs[0]-max_z)
        sum_exp_z = exp_z.sum(axes=self.axes)

        if self.axes is None:
            new_shape = tuple([1] * len(node.inputs[0].shape))
            return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape) * exp_z / sum_exp_z
        new_shape = list(node.inputs[0].shape)
        if isinstance(self.axes, int):
            self.axes = (self.axes,)
        for axis in self.axes:
            new_shape[axis] = 1
        out_grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        sum_exp_z = sum_exp_z.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        return out_grad * exp_z / sum_exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

