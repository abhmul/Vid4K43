import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Layer, Conv2d, BatchNorm2d, PixelShuffle_ICNR, SelfAttention
from . import utils


def concat(inputs):
    return torch.cat(inputs, dim=1)


class ResidualBlock(Layer):
    def __init__(
        self,
        kernel_size,
        input_shape=None,
        activation=None,
        batchnorm=False,
        spectral_norm=False,
        dense=False,
        bottle=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.dense = dense
        self.bottle = bottle
        self.spectral_norm = spectral_norm
        self.input_shape = input_shape

        self.conv1 = lambda input_filters: Conv2d(
            input_filters // 2 if bottle else input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
            spectral_norm=self.spectral_norm,
        )
        self.conv2 = lambda input_filters: Conv2d(
            input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
            spectral_norm=self.spectral_norm,
        )
        self.merge = concat if dense else sum

        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        self.input_shape = utils.get_image_shape(inputs)
        input_filters = utils.get_num_channels(inputs)
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
        self.bn = BatchNorm2d()
        self.conv = Conv2d(
            self.output_filters,
            kernel_size=3,
            activation=nn.ReLU(),
            spectral_norm=self.spectral_norm,
        )
        # self.att = SelfAttention() if self.self_attention else lambda x: x
        if self.self_attention:
            self.att = SelfAttention()
        self.merge = concat

    def forward(self, upsample_input, residual_input):
        upsample_output = self.upsampler(upsample_input)
        x_img_shape = utils.get_image_shape(residual_input)
        upsample_img_shape = utils.get_image_shape(upsample_output)
        if x_img_shape != upsample_img_shape:
            upsample_output = F.interpolate(
                upsample_output, x_img_shape, mode="nearest"
            )
        cat_x = self.merge([upsample_output, self.bn(residual_input)])
        x = self.conv(cat_x)
        if self.self_attention:
            x = self.att(x)
        return x

