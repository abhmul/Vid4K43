import pytest

from glob import glob
from kaggleutils import plot_img_lists

from data import *

TEST_IMAGES = "test_imgs/*.jpg"


def test_crappify_dataset():
    crop_size = (256, 256)
    crappifier = Crappify(0.5)
    data = CrappifyDataset(TEST_IMAGES, crappifier=crappifier, crop_size=crop_size)
    imagepaths = sorted(glob(TEST_IMAGES))
    assert len(data) == len(imagepaths)
    data_x, data_y = [], []
    for i, (x, y) in enumerate(data):
        assert (
            x.size == y.size
        ), f"Sample size x: {x.size} - y: {y.size} does not match for sample {data.samples[i]}"
        assert (
            x.size == crop_size
        ), f"Sample size {x.size} does not match crop size {crop_size} for sample {data.samples[i]}"
        data_x.append(x)
        data_y.append(y)
    assert len(data_x) == len(imagepaths)

    # Plot for human evaluation
    plot_img_lists(data_x, data_y)
