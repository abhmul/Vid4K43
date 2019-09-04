from kaggleutils import get_random_seed

TRAINING_CONFIG = dict(
    encoder="resnet18",
    self_attention=True,
    channels_factor=1,
    batchnorm=False,
    spectral_norm=True,
    batch_size=32,
    epoch_size=1000,
    epochs=10,
    crop_size=(256, 256)
    seed=get_random_seed(),
)
