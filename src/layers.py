import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from pyjet.layers import (
    Layer,
    Conv2D,
    BatchNorm2D,
    Conv1D,
    Identity,
    Input,
    Concatenate,
    Add,
)
import pyjet.layers.layer_utils as utils
from pyjet.models import SLModel
import pyjet.backend as J

assert J.channels_mode == "channels_first", "Must use J.channels_mode='channels_first'"


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init of `x`, with `scale` and `init` function.
    TODO: Read more about this."""
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(J.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


# The fastai PixelShuffle uses NormType.weight and no BN
# https://github.com/fastai/fastai/blob/master/fastai/layers.py#L192
class PixelShuffle_ICNR(Layer):
    "Upsample by `scale` from input filters to output filters, using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
        self, output_filters=None, scale=2, input_shape=None, spectral_norm=False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_filters = output_filters
        self.scale = scale
        self.spectral_norm = spectral_norm

        self.conv = lambda filters: Conv2D(
            filters * (scale ** 2),
            kernel_size=1,
            input_shape=self.input_shape,
            spectral_norm=self.spectral_norm,
        )
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = F.relu

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        # Calling it once builds the layer
        if self.output_filters is None:
            self.output_filters = utils.get_channels(inputs)
        self.conv = self.conv(self.output_filters)
        x = self.conv(inputs)
        icnr(self.conv.weight())

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


class ResidualBlock(Layer):
    def __init__(
        self,
        kernel_size,
        input_shape=None,
        activation="linear",
        batchnorm=False,
        dense=False,
        bottle=False,
        spectral_norm=False,
    ):
        """There are some discrepancies between this implementation and Fastai's res_block, look here if issues"""
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.dense = dense
        self.bottle = bottle
        self.spectral_norm = spectral_norm
        self.input_shape = input_shape

        self.conv1 = lambda input_filters: Conv2D(
            input_filters // 2 if bottle else input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
            spectral_norm=self.spectral_norm,
        )
        self.conv2 = lambda input_filters: Conv2D(
            input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
            spectral_norm=self.spectral_norm,
        )
        self.merge = Concatenate(dim=0) if dense else Add()

        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        input_filters = utils.get_channels(inputs)
        self.conv1 = self.conv1(input_filters)
        self.conv2 = self.conv2(input_filters)

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.merge([x, orig])


class UnetBlockWide(Layer):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        output_filters: int,
        self_attention=False,
        input_shape=None,
        upsample_input_shape=None,
        spectral_norm=False,
    ):
        super().__init__()
        self.output_filters = output_filters
        self.self_attention = self_attention
        self.input_shape = input_shape
        self.upsample_input_shape = upsample_input_shape
        self.spectral_norm = spectral_norm

        self.upsample_output_filters = self.output_filters
        self.upsampler = PixelShuffle_ICNR(self.upsample_output_filters)
        self.bn = BatchNorm2D()
        self.conv = Conv2D(
            self.output_filters,
            kernel_size=3,
            input_activation="relu",
            spectral_norm=self.spectral_norm,
        )
        self.att = SelfAttention() if self.self_attention else Identity
        self.merge = Concatenate(dim=0)

    def forward(self, upsample_input, residual_input):
        upsample_output = self.upsampler(upsample_input)
        x_img_shape = utils.get_shape_no_channels(residual_input)
        upsample_img_shape = utils.get_shape_no_channels(upsample_output)
        if x_img_shape != upsample_img_shape:
            upsample_output = F.interpolate(
                upsample_output, x_img_shape, mode="nearest"
            )
        cat_x = self.merge((upsample_output, self.bn(residual_input)))
        return self.att(self.conv(cat_x))


class SelfAttention(Layer):
    "Self attention layer for nd."

    def __init__(self, input_shape=None, spectral_norm=False):
        super().__init__()
        self.input_shape = input_shape
        self.spectral_norm = spectral_norm
        # Save them as constructors and we'll build them in the builder
        self.f = lambda input_channels: Conv2D(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.g = lambda input_channels: Conv2D(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.h = lambda input_channels: Conv2D(
            input_channels // 8, kernel_size=1, spectral_norm=self.spectral_norm
        )
        # jantic left this out of his code, probably because a complete c x c layer
        # would have been able to encode the v-h combo and more. But the way from
        # the paper (include v and h) should be fewer parameters.
        self.v = lambda input_channels: Conv2D(
            input_channels, kernel_size=1, spectral_norm=self.spectral_norm
        )
        self.gamma = nn.Parameter(J.tensor(0.0))

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        # Calling it once builds the layer
        input_shape = utils.get_input_shape(inputs)
        input_channels = utils.get_channels(inputs)
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


# TODO: Move this to PyJet
class Conv2DScaleChannels(Layer):
    def __init__(self, scale=1, **conv_kwargs):
        super().__init__()
        self.scale = scale
        self.conv = lambda input_channels: Conv2D(
            input_channels * self.scale, **conv_kwargs
        )

        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        input_channels = utils.get_channels(inputs)
        self.conv = self.conv(input_channels)

    def forward(self, inputs):
        return self.conv(inputs)


# TODO: Spectral norm type
class DynamicUnetWide(Layer):
    """Create a U-Net from a given architecture."""

    # Assume the input channels is 3
    input_channels = 3
    net_base_channels = 256

    def __init__(
        self, encoder, channels_factor=1, batchnorm=False, spectral_norm=False
    ):
        """`encoder` must be a resnet"""
        super().__init__()
        # First check if we need to cast the encoder to cuda
        if J.use_cuda:
            encoder = encoder.cuda()

        self.channels_factor = channels_factor
        self.channels = self.net_base_channels * self.channels_factor
        self.batchnorm = batchnorm
        self.spectral_norm = spectral_norm

        # Define the network
        self.encoder = encoder
        self.neck = nn.Sequential(
            Conv2DScaleChannels(
                scale=2,
                kernel_size=3,
                activation="relu",
                batchnorm=self.batchnorm,
                spectral_norm=self.spectral_norm,
            ),
            Conv2DScaleChannels(
                scale=1,
                kernel_size=3,
                activation="relu",
                batchnorm=self.batchnorm,
                spectral_norm=self.spectral_norm,
            ),
        )

        self.unet_layers = []
        # We'll use a little different logic for the last 2 ones
        assert self.encoder.num_residuals >= 2  # This includes the input
        for _ in range(self.encoder.num_residuals - 2):
            # In his code he only uses self attention on the 3rd to last layer
            # We'll try it everywhere and come back and fix if it's not working
            unet_block = UnetBlockWide(
                self.channels, self_attention=True, spectral_norm=self.spectral_norm
            )
            self.unet_layers.append(unet_block)

        # And the penultimate one
        unet_block = UnetBlockWide(
            self.channels // 2, self_attention=False, spectral_norm=self.spectral_norm
        )
        self.unet_layers.append(unet_block)
        self.unet_layers = nn.ModuleList(self.unet_layers)

        # And the final one
        self.upsampler = PixelShuffle_ICNR(scale=2, spectral_norm=self.spectral_norm)
        self.merge = Concatenate(dim=0)
        self.res_block = ResidualBlock(
            kernel_size=3,
            activation="relu",
            batchnorm=self.batchnorm,
            dense=False,
            spectral_norm=self.spectral_norm,
        )
        # Final Conv layer
        self.final_conv = Conv2D(
            self.input_channels, kernel_size=1, activation="linear"
        )

    def forward(self, x):
        # Encoder
        x, residuals = self.encoder(x)
        # Neck
        x = self.neck(x)
        # Unets - save the last residual for the last cross
        for residual_input, unet_layer in zip(
            reversed(residuals[1:]), self.unet_layers
        ):
            residual_input = residual_input
            x = unet_layer(x, residual_input)
        # last cross
        x = self.upsampler(x)
        x = self.merge([x, residuals[0]])
        x = self.res_block(x)
        return self.final_conv(x)

