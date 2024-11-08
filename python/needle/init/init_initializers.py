import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):

    return rand(fan_in, fan_out, 
                low = -math.sqrt(6 / (fan_in + fan_out)) * gain, 
                high = math.sqrt(6 / (fan_in + fan_out)) * gain, 
                **kwargs)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):

    return randn(fan_in, fan_out, 
                mean = 0, 
                std = math.sqrt(2 / (fan_in + fan_out)) * gain, 
                **kwargs)

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"

    gain = math.sqrt(2)
    if shape:
        return rand( *shape, 
                low = -math.sqrt(3 / fan_in) * gain, 
                high = math.sqrt(3 / fan_in) * gain, 
                **kwargs)
    else:
        return rand( fan_in, fan_out, 
                low = -math.sqrt(3 / fan_in) * gain, 
                high = math.sqrt(3 / fan_in) * gain, 
                **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"

    gain = math.sqrt(2)
    return randn( fan_in, fan_out, 
                mean = 0, 
                std = gain / math.sqrt(fan_in), 
                **kwargs)
