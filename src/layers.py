import numpy as np

import torch
import torch.nn as nn
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
from pyjet.hooks import hook_outputs, model_sizes
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

    def __init__(self, output_filters: int, scale: int = 2, input_shape=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_filters = output_filters
        self.scale = scale

        self.conv = Conv2D(
            output_filters * (scale ** 2), kernel_size=1, input_shape=self.input_shape
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
    ):
        """There are some discrepancies between this implementation and Fastai's res_block, look here if issues"""
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.dense = dense
        self.bottle = bottle
        self.input_shape = input_shape

        self.conv1 = lambda input_filters: Conv2D(
            input_filters // 2 if bottle else input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
        )
        self.conv2 = lambda input_filters: Conv2D(
            input_filters,
            self.kernel_size,
            activation=self.activation,
            batchnorm=self.batchnorm,
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
    ):
        super().__init__()
        self.output_filters = output_filters
        self.self_attention = self_attention
        self.input_shape = input_shape
        self.upsample_input_shape = upsample_input_shape

        self.upsample_output_filters = self.output_filters
        self.upsampler = PixelShuffle_ICNR(self.upsample_output_filters)
        self.bn = BatchNorm2D()
        self.conv = Conv2D(self.output_filters, kernel_size=3, input_activation="relu")
        self.att = SelfAttention() if self.self_attention else Identity
        self.merge = Concatenate(dim=0)

    def forward(self, residual_input, upsample_input):
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

    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape
        # Save them as constructors and we'll build them in the builder
        self.query = lambda input_channels: Conv1D(input_channels, input_channels // 8)
        self.key = lambda input_channels: Conv1D(input_channels, input_channels // 8)
        self.value = lambda input_channels: Conv1D(input_channels, input_channels)
        self.gamma = nn.Parameter(J.tensor(0.0))

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        # Calling it once builds the layer
        input_shape = utils.get_input_shape(inputs)
        input_channels = utils.get_channels(inputs)
        self.query = self.query(input_channels)
        self.key = self.key(input_channels)
        self.value = self.value(input_channels)

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        # Flatten x along the length, width dimension
        size = x.size()
        x = x.view(*size[:2], -1)
        # Compute the q, k, and v
        q, k, v = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(q.permute(0, 2, 1).contiguous(), k), dim=1)
        # Combine with input and reshape
        o = self.gamma * torch.bmm(v, beta) + x
        return o.view(*size).contiguous()


# TODO: Spectral norm type
class DynamicUnetWide(SLModel):
    """Create a U-Net from a given architecture."""

    def __init__(self, encoder, channels_factor=1, input_channels=3):
        super().__init__()
        channels = 256 * channels_factor
        # We'll use this for probing the model for output sizes
        dummy_image_size = (input_channels, 256, 256)
        dummy_input = Input(dummy_image_size)
        encoder.eval()  # Make this eval while we're probing it
        # Hook the outputs for skip connections we want
        self.skip_idxs, self.skip_sizes = self.get_skip_idxs_sizes(encoder, dummy_input)
        self.skip_outputs = hook_outputs(
            [encoder[i] for i in self.skip_idxs], detach=False
        )
        # Run a dummy input through the encoder to get the output channel size.
        encoder_out = encoder(dummy_input).detach()
        encoder_out_channels = utils.get_channels(encoder_out)
        encoder.train()

        # Define the network
        self.encoder = encoder
        self.encoder_batchnorm = BatchNorm2D()
        self.encoder_activation = nn.ReLU()
        self.bottleneck = nn.Sequential(
            Conv2D(
                encoder_out_channels * 2,
                kernel_size=3,
                activation="relu",
                batchnorm=True,
            ),
            Conv2D(
                encoder_out_channels, kernel_size=3, activation="relu", batchnorm=True
            ),
        )

        self.unet_layers = []
        # We'll use a little different logic for the last one
        for _ in self.skip_idxs[:-1]:
            # In his code he only uses self attention on the 3rd to last layer
            # We'll try it everywhere and come back and fix if it's not working
            unet_block = UnetBlockWide(channels, self_attention=True)
            self.unet_layers.append(unet_block)
        # And the final one
        unet_block = UnetBlockWide(channels // 2, self_attention=False)
        self.unet_layers.append(unet_block)
        self.unet_layers = nn.ModuleList(self.unet_layers)

        # If our image does not match up we need to do some resizing
        if dummy_image_size[1:] != self.skip_sizes[-1][1:]:
            print(
                f"Image size {dummy_image_size} does not match current output size {self.skip_sizes[-1]}"
            )
            # If this happens we need to do a PixelShuffle
            raise ValueError

        # The last cross gets the original input and merges with the current output
        self.merge = Concatenate(dim=0)
        self.res_block = ResidualBlock(
            kernel_size=3, activation="relu", batchnorm=True, dense=False
        )

        # Final Conv layer
        self.final_conv = Conv2D(input_channels, kernel_size=1, activation="linear")

        self.infer_inputs(dummy_input)

    def get_skip_idxs_sizes(self, encoder, dummy_input):
        "Get the indexes of the layers where the size of the activation changes."
        encoder_sizes = model_sizes(encoder, dummy_input)
        # We're using "channels_first"
        skip_channels = np.array([size[0] for size in encoder_sizes])
        skip_idxs = list(reversed(np.where(skip_channels[:-1] != skip_channels[1:])[0]))
        skip_sizes = [encoder_sizes[i] for i in skip_idxs]
        print(f"Skip sizes are {skip_sizes}")
        return skip_idxs, skip_sizes

    def forward(self, x):
        orig = x
        # Encoder
        x = self.encoder(x)
        x = self.encoder_batchnorm(x)
        x = self.encoder_activation(x)
        # Neck
        x = self.bottleneck(x)
        # Unets
        for residual_input, unet_layer in zip(self.skip_outputs, self.unet_layers):
            residual_input = residual_input.stored[0]  # There should only be one item
            print(residual_input.size())  # For debugging
            x = unet_layer(residual_input, x)
        # last cross
        x = self.merge([x, orig])
        x = self.res_block(x)
        self.loss_in = self.final_conv(x)

        return torch.sigmoid(self.loss_in)

