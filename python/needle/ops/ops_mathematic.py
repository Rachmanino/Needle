"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return out_grad * self.scalar * (input ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idx = list(range(len(a.shape)))
        if self.axes is None:
            idx[-1], idx[-2] = idx[-2], idx[-1]
        else:
            assert len(self.axes) == 2
            idx[self.axes[0]], idx[self.axes[1]] = idx[self.axes[1]], idx[self.axes[0]]
        return a.permute(tuple(idx))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(out_grad.shape, node.inputs[0].shape)
        if out_grad.shape == node.inputs[0].shape:
            return out_grad
        new_axes = list(range(len(self.shape)))
        old_dims, new_dims = node.inputs[0].shape, self.shape
        for i in range(len(self.shape)):
            if old_dims[i] == new_dims[i]:
                new_axes[i] = -1
        new_axes = tuple(filter(lambda x: x >= 0, new_axes))
        return out_grad.sum(axes=new_axes).reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ### Support sum over multiple axes
        if isinstance(self.axes, tuple) and len(self.axes) > 1:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        return a.sum(axis = self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:

            return out_grad.reshape(tuple([1] * len(node.inputs[0].shape))) \
                    .broadcast_to(node.inputs[0].shape)
        new_shape = list(node.inputs[0].shape)
        if isinstance(self.axes, int):
            new_shape[self.axes] = 1
        else:
            for axis in self.axes:
                new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a, grad_b = out_grad @ transpose(b), transpose(a) @ out_grad
        
        if len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape) - len(a.shape))))

        if len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape) - len(b.shape))))
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0].cached_data > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad -  out_grad * tanh(node.inputs[0]) ** 2
        #! 这里写成out_grad * (1 - tanh(node.inputs[0]) ** 2)过不了
        #! test_tanh_backward, 很奇怪
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array"
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        new_shape = tuple(new_shape)
        output = array_api.empty(new_shape, device=args[0].device)
        indice = [slice(0, s) for s in new_shape]
        for i, a in enumerate(args):
            assert shape == a.shape, "All arrays need to be of the same size"
            indice[self.axis] = slice(i, i+1)
            output[tuple(indice)] = a
        return output
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        output = []
        indices = [slice(0, s) for s in A.shape]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            indices[self.axis] = slice(i, i+1)
            output.append(A[tuple(indices)].compact().reshape(new_shape))
        return tuple(output)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        indices = [slice(0, s) for s in a.shape]
        for axis in self.axes:
            new_shape[axis] *= (self.dilation+1)
            indices[axis] = slice(0, new_shape[axis], self.dilation+1)
        new_shape, indices = tuple(new_shape), tuple(indices)
        output = a.device.full(new_shape, 0)
        output[indices] = a
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        indices = [slice(0, s) for s in a.shape]
        for axis in self.axes:
            indices[axis] = slice(0, a.shape[axis], self.dilation+1)
        indices = tuple(indices)
        output = a[indices]
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, 
                A,  # img, (N, H, W, Cin) after padding
                B   # kernel, (K, K, Cin, Cout)
    ): # output, (N, H', W', Cout)
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0),
                  (self.padding, self.padding),
                  (self.padding, self.padding),
                  (0, 0)))
        
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape
        Ns, Hs, Ws, Cins = A.strides
        inner_dim = Cin * K * K
        out_H = (H - K + 1) // self.stride
        out_W = (W - K + 1) // self.stride

        im2col = A.as_strided(shape = (N, out_H, out_W, K, K, Cin),
                              strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cins)).compact().reshape((N * out_H * out_W, inner_dim))
        kernel = B.compact().reshape((inner_dim, Cout))
        return (im2col @ kernel).compact().reshape((N, out_H, out_W, Cout))   
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        '''
            Conv: (N, H, W, Cin) @ (K, K, Cin, Cout) = (N, H', W', Cout)
        '''
        A, B = node.inputs 
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1)
        gradA = conv(out_grad, # (N, H', W', Cout)
                     flip(B, axes=(0, 1)).transpose((2,3)), # (K, K, Cout, Cin)
                     padding = K-1-self.padding)
        gradB = conv(A.transpose((0, 3)), # (Cin, H, W, N) 
                     out_grad.transpose((0, 1)).transpose((1, 2)), # (H+2P-K+1, W+2P-K+1, N, Cout)
                     padding = self.padding).transpose((0, 1)).transpose((1, 2)) 
        return gradA, gradB
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
