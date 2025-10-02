import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import math

def initializer_conv(shape, init='he', dist='truncnorm', dist_scale=1.0):
    w_width = shape[3]
    w_height = shape[2]
    size_in = shape[1]
    size_out = shape[0]
    limit = 0.

    if init == 'xavier':
        limit = math.sqrt(2. / (w_width * w_height * (size_in + size_out))) * dist_scale
    elif init == 'he':
        limit = math.sqrt(2. / (w_width * w_height * size_in)) * dist_scale
    else:
        raise Exception('Arg `init` not recognized.')

    if dist == 'norm':
        var = np.array(stats.norm(loc=0, scale=limit).rvs(shape)).astype(np.float32)
    elif dist == 'truncnorm':
        var = np.array(stats.truncnorm(a=-2, b=2, scale=limit).rvs(shape)).astype(np.float32)
    elif dist == 'uniform':
        var = np.array(stats.uniform(loc=-limit, scale=2*limit).rvs(shape)).astype(np.float32)
    else:
        raise Exception('Arg `dist` not recognized.')

    return var

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 init='he', dist='truncnorm', dist_scale=1.0):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.weight = nn.Parameter(torch.Tensor(
            initializer_conv([out_channels, in_channels // groups, kernel_size, kernel_size],
                             init=init, dist=dist, dist_scale=dist_scale))
        )
