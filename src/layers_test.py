from pyjet.layers import layer_utils as utils

from resnet import *
from layers import *
import pytest


def test_pixelshuffle():
    x = PixelShuffle_ICNR(10)(Input(3, 40, 50))
    assert utils.get_input_shape(x) == (10, 80, 100)


def test_residualblock():
    x = ResidualBlock(kernel_size=3, dense=False)(Input(10, 40, 50))
    assert utils.get_input_shape(x) == (10, 40, 50)
    x = ResidualBlock(kernel_size=3, dense=True)(Input(10, 40, 50))
    assert utils.get_input_shape(x) == (20, 40, 50)


def test_unetblockwide():
    x = UnetBlockWide(10)(Input(3, 20, 25), Input(3, 40, 50))
    assert utils.get_input_shape(x) == (10, 40, 50)


def test_selfattention():
    # The input channels must be greater than 8
    orig_shape = (8, 40, 50)
    x = SelfAttention()(Input(*orig_shape))
    assert utils.get_input_shape(x) == orig_shape


def test_conv2dchannelscale():
    orig_shape = (8, 40, 50)
    x = Conv2DScaleChannels(2, kernel_size=3)(Input(*orig_shape))
    assert orig_shape[0] * 2 == utils.get_channels(x)


def test_dynamicunetwide():
    orig_shape = (3, 256, 256)
    dummy_input = Input(*orig_shape)
    encoder = resnet34()
    x = DynamicUnetWide(encoder, channels_factor=1)(dummy_input)
    assert orig_shape == utils.get_input_shape(x)

