import logging
import torch.nn as nn
import torch.nn.functional as F

from . import Layer
from . import utils


class Conv2d(Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        activation=None,
        batchnorm=False,
        spectral_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        if padding != "same" and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm
        self.spectral_norm = spectral_norm
        self.dropout = dropout

        self.in_channels = None
        self.conv_layers = None
        self.conv = None
        self._padding_h = None
        self._padding_w = None
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        self.input_shape = utils.get_sample_shape(inputs)
        self.in_channels = inputs.size(1)
        h, w = inputs.size(2), inputs.size(3)
        self._padding_h = self.get_padding(h)
        self._padding_w = self.get_padding(w)
        self.conv_layers = nn.Sequential()

        # Add the conv layer
        conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )
        if self.spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        self.conv_layers.add_module(name="conv", module=conv)
        self.conv = conv
        self.weight = self.conv.weight

        # Add the other pieces
        if self.batchnorm:
            self.batchnorm = nn.BatchNorm2d(self.out_channels)
            self.conv_layers.add_module(name="bn", module=self.batchnorm)
        if self.activation is not None:
            self.conv_layers.add_module(name="activation", module=self.activation)
        if self.dropout:
            self.dropout = nn.Dropout2d(self.dropout)
            self.conv_layers.add_module(name="dropout", module=self.dropout)

    def pad_input(self, inputs):
        # inputs is batch_size x channels x height x width
        padding = self.get_padding(inputs.size(2)) + self.get_padding(inputs.size(3))
        return F.pad(inputs, padding)

    def get_padding(self, input_len):
        if self.padding != "same":
            return self.padding, self.padding
        else:
            return self.get_same_padding(input_len)

    def get_same_padding(self, input_len):
        total_padding = int(
            self.stride * (input_len - 1)
            + 1
            + self.dilation * (self.kernel_size - 1)
            - input_len
        )
        pad_l = total_padding // 2
        return pad_l, total_padding - pad_l

    def forward(self, inputs):
        return self.conv_layers(self.pad_input(inputs))

    def __str__(self):
        return "%r" % self.conv_layers


class Conv2dScaleChannels(Layer):
    def __init__(self, scale=1, **conv_kwargs):
        super().__init__()
        self.scale = scale
        self.conv = lambda input_channels: Conv2d(
            input_channels * self.scale, **conv_kwargs
        )

        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        input_channels = utils.get_num_channels(inputs)
        self.conv = self.conv(input_channels)

    def forward(self, inputs):
        return self.conv(inputs)
