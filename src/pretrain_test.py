from pyjet.layers import Input

from pretrain import *
import matplotlib.pyplot as plt

from kaggleutils import plot_img_lists

DATA_GLOB = "/media/abhmul/BackupSSD1/Datasets/open-image/images/*"


def test_prepare_model():
    model = prepare_model(
        # Model Params
        encoder="resnet18",
        channels_factor=1,
        batchnorm=False,
        spectral_norm=True,
        # Loss Params
        layer_weights=[20, 70, 10],
        # Optimizer Params
        lr_range=(1e-7, 1e-3),
        momentum_range=(0.95, 0.85),
        period=10,
    )

    print(repr(model))

    orig_shape = (3, 256, 256)
    dummy_input = Input(*orig_shape)
    out = model.cast_output_to_numpy(model(dummy_input))
    plot_img_lists(out.transpose(0, 2, 3, 1))


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

        plot_img_lists(
            batch_tr_x.transpose(0, 2, 3, 1), batch_tr_y.transpose(0, 2, 3, 1)
        )

