TRAINING_CONFIG = dict(
    # Generator Params
    encoder="resnet18",
    self_attention=True,
    channels_factor=1,
    batchnorm=False,
    spectral_norm=True,
    # Loss Params
    layer_weights=[20, 70, 10],
    # Optimizer Params
    lr_range=(1e-7, 1e-4),
    momentum_range=(0.95, 0.85),
    period=8 * ((1000 + 32 - 1) // 32),  # 8 * steps_per_epoch
    # Data Params
    test_split=0.1,
    # Training params
    batch_size=32,
    epoch_size=1000,
    epochs=20,
    crop_size=(256, 256),
    seed=259,
)
