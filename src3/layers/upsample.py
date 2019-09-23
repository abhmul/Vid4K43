import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Layer, Conv2d
from . import utils


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init of `x`, with `scale` and `init` function.
    TODO: Read more about this."""
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


# The fastai PixelShuffle uses NormType.weight and no BN
# https://github.com/fastai/fastai/blob/master/fastai/layers.py#L192
class PixelShuffle_ICNR(Layer):
    "Upsample by `scale` from input filters to output filters, using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
        self, out_channels=None, scale=2, input_shape=None, spectral_norm=False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.scale = scale
        self.spectral_norm = spectral_norm

        self.conv = lambda filters: Conv2d(
            filters * (scale ** 2),
            kernel_size=1,
            input_shape=self.input_shape,
            spectral_norm=self.spectral_norm,
            activation=nn.ReLU(),
        )
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        # Calling it once builds the layer
        self.input_shape = utils.get_sample_shape(inputs)
        if self.out_channels is None:
            self.out_channels = utils.get_num_channels(inputs)
        self.conv = self.conv(self.out_channels)
        x = self.conv(inputs)
        icnr(self.conv.weight)

    def forward(self, x):
        x = self.shuf(self.conv(x))
        return self.blur(self.pad(x))
