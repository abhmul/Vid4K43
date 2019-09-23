import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Layer, Conv2d
from . import utils


class SelfAttention(Layer):
    "Self attention layer for nd."

    def __init__(self, input_shape=None, spectral_norm=False):
        super().__init__()
        self.input_shape = input_shape
        self.spectral_norm = spectral_norm
        # Save them as constructors and we'll build them in the builder
        self.f = lambda input_channels: Conv2d(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.g = lambda input_channels: Conv2d(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.h = lambda input_channels: Conv2d(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        # jantic left this out of his code, probably because a complete c x c layer
        # would have been able to encode the v-h combo and more. But the way from
        # the paper (include v and h) should be fewer parameters.
        self.v = lambda input_channels: Conv2d(
            input_channels, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        # Calling it once builds the layer
        input_shape = utils.get_sample_shape(inputs)
        input_channels = utils.get_num_channels(inputs)
        self.f = self.f(input_channels)
        self.g = self.g(input_channels)
        self.h = self.h(input_channels)
        self.v = self.v(input_channels)

    def flatten_img(self, x):
        shape = tuple(x.size())
        return x.view(*shape[:2], -1), shape

    def unflatten_img(self, x, shape):
        return x.view(*shape)

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        # Flatten x along the length, width dimension
        # Compute the f, k, and v

        f, g, h = self.f(x), self.g(x), self.h(x)
        f, fshape = self.flatten_img(f)
        g, gshape = self.flatten_img(g)
        h, hshape = self.flatten_img(h)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        # Combine with input and reshape
        inner = torch.bmm(h, beta)
        target_shape = (inner.size(0), inner.size(1), hshape[2], hshape[3])
        inner = self.unflatten_img(inner, target_shape)
        o = self.v(inner)
        y = self.gamma * o + x
        return y
