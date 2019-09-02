import torch
import torch.nn as nn
import torch.nn.functional as F
from pyjet.layers import Layer, Conv2D, BatchNorm2D
import pyjet.layers.layer_utils as utils
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
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = F.relu

    @utils.builder
    def build_layer(self, inputs):
        # Calling it once builds the layer
        x = self.conv(inputs)
        icnr(self.conv.weight())

    def forward(self, x):
        if not self.built:
            print("This should just show up once.")
            self.build_layer(x)
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


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

    def forward(self, x_input, upsample_input):
        upsample_output = self.upsampler(upsample_input)
        x_img_shape = utils.get_shape_no_channels(x_input)
        upsample_img_shape = utils.get_shape_no_channels(upsample_output)
        if x_img_shape != upsample_img_shape:
            upsample_output = F.interpolate(
                upsample_output, x_img_shape, mode="nearest"
            )
        cat_x = torch.cat((upsample_output, self.bn(x_input)), dim=1))
        return self.conv(cat_x)
