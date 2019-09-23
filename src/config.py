# No first residual + epoch size 1k Loss: ~1.48 Train 1.39 Val
epoch_size = 10000
batch_size = 16
crop_size_side = 128

TRAINING_CONFIG = dict(
    # Generator Params
    encoder="resnet18",
    self_attention=True,
    channels_factor=1,
    batchnorm=False,
    spectral_norm=True,
    input_batchnorm=True,
    # Loss Params
    layer_weights=[20, 70, 10],
    # Optimizer Params
    lr_range=(1e-5, 1e-2),
    momentum_range=(0.95, 0.85),
    period_factor=5,
    # Data Params
    test_split=0.1,
    resize_factor=0.5,  # Set to 1 to turn off
    noise=True,
    # Training params
    batch_size=batch_size,
    epoch_size=epoch_size,
    epochs=100,
    crop_size=(crop_size_side, crop_size_side),
    seed=259,
    # Test Params
    test_batch_size=1,
    test_crop_size=(512, 512),
    # Model path
    fname="gen_pretrain_05_noise",
)
