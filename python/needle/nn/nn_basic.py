"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, 
                                                     device=device,
                                                     requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, 
                                                       device=device,
                                                       requires_grad=True).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = X @ self.weight
        if self.bias is not None:
            output += self.bias.broadcast_to(output.shape)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = 1
        for dim in X.shape:
            size *= dim
        return X.reshape((X.shape[0], size // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Tanh(Module):
    '''
        My implementation for Tanh activation, 
        which is not required in the assignment.
    '''
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch = logits.shape[0]
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, requires_grad=True) #! 注意用法，第一个参数是num_classes
        a = ops.logsumexp(logits, (1,))
        output = (a / logits.shape[0]).sum() - (y_one_hot * logits / logits.shape[0]).sum()
        return output
        ### END YOUR SOLUTION
        #! 这里为什么第二种做法dtype会变成float64呢?


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device)
        self.running_var = init.ones(self.dim, device=device)
        self.device = device
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        if self.training:
            avg = x.sum(axes=0) / batch
            var = ((x - avg.reshape((1, self.dim)).broadcast_to(x.shape)) ** 2).sum(axes=0) / batch
            self.running_mean = self.running_mean * (1 - self.momentum) + avg * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum
            return (x - avg.reshape((1, self.dim)).broadcast_to(x.shape)) / (var.reshape((1, self.dim)).broadcast_to(x.shape) + self.eps) ** 0.5 * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
            return (x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / (self.running_var.reshape((1, self.dim)).broadcast_to(x.shape) + self.eps) ** 0.5 * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module): #! not tested
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, requires_grad=True))
        self.device = device
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[-1] == self.dim
        tmp_shape = list(x.shape)
        tmp_shape[-1] = 1
        tmp_shape = tuple(tmp_shape)
        tmp_param_shape = [1] * len(x.shape)
        tmp_param_shape[-1] = self.dim
        tmp_param_shape = tuple(tmp_param_shape)

        avg = x.sum(axes=len(x.shape)-1).reshape(tmp_shape).broadcast_to(x.shape) / self.dim
        var = ((x - avg) ** 2).sum(axes=len(x.shape)-1).reshape(tmp_shape).broadcast_to(x.shape) / self.dim
        return (x - avg) / ((var + self.eps) ** 0.5) * self.weight.reshape(tmp_param_shape).broadcast_to(x.shape) + self.bias.reshape(tmp_param_shape).broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, 
                                  p=1-self.p, 
                                  device = x.device,
                                  dtype = x.dtype) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
