from pretrain import *
from data import ImageNetNormalizer
import matplotlib.pyplot as plt

from kaggleutils import plot_img_lists

DATA_GLOB = "/media/abhmul/BackupSSD1/Datasets/open-image/images/*"


def test_prepare_data():
    batch_size = 32
    crop_size = (256, 256)
    tr, v, te = prepare_data(
        DATA_GLOB, batch_size=batch_size, epoch_size=100, crop_size=crop_size
    )

    assert tr.steps_per_epoch == 4
    assert v.steps_per_epoch == 1
    assert te.steps_per_epoch == 1
    for gen in (tr, v, te):
        batch_tr_x, batch_tr_y = next(gen)
        for batch_tr_imgs in (batch_tr_x, batch_tr_y):
            assert len(batch_tr_imgs) == batch_size
            assert batch_tr_imgs.shape[1] == 3  # Channels
            assert batch_tr_imgs.shape[2:] == crop_size

        batch_tr_x = ImageNetNormalizer.unnormalize_float(batch_tr_x)
        batch_tr_y = ImageNetNormalizer.unnormalize_float(batch_tr_y)
        plot_img_lists(
            batch_tr_x.transpose(0, 2, 3, 1), batch_tr_y.transpose(0, 2, 3, 1)
        )

