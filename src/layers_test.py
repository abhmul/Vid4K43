from pyjet.layers import layer_utils as utils

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
    x = UnetBlockWide(10)(Input(3, 40, 50), (Input(3, 20, 25)))
    assert utils.get_input_shape(x) == (10, 40, 50)


def test_selfattention():
    orig_shape = (3, 40, 50)
    x = SelfAttention()(Input(*orig_shape))
    assert utils.get_input_shape(x) == orig_shape

