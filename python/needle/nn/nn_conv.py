"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU, BatchNorm2d


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = (kernel_size - 1) // 2 # to ensure padding=same given stride=1
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_channels * kernel_size**2, 
            fan_out=out_channels * kernel_size**2,
            shape=(kernel_size, kernel_size, in_channels, out_channels), 
            device=device, 
            dtype=dtype,
            requires_grad=True))
        if bias:
            self.bias = Parameter(init.rand(
                out_channels,
                low=-1.0/(in_channels * kernel_size**2)**0.5,
                high=1.0/(in_channels * kernel_size**2)**0.5,
                device=device, 
                dtype=dtype,
                requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:

        nhwc_x = x.transpose((1, 2)).transpose((2, 3))
        nhwc_output = ops.conv(nhwc_x, 
                               self.weight, 
                               stride = self.stride, 
                               padding = self.padding)
        if self.bias:
            nhwc_output += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(nhwc_output.shape)
        return nhwc_output.transpose((2, 3)).transpose((1, 2))

class ConvBN(Module):
    '''
        My implementation for ConvBN block, which consists of Conv, BatchNorm2d and ReLU
        Note that this module is not required in the assignment and is only for my own convenience
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):   
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype)
        self.bn = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x