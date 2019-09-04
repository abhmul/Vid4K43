from glob import glob
import argparse

import numpy as np
from kaggleutils import set_random_seed
from pyjet.data import ImageDataset

from resnet import *
from config import TRAINING_CONFIG as TC
from data import RandomCropper, CrappifyTransformer, ImageNetNormalizer
from loss import FeatureLoss
from model import Generator


parser = argparse.ArgumentParser(description="Pretrain the 4kifier with real images")
parser.add_argument("-d", "--data", help="Glob for image data.")

set_random_seed(TC["seed"])


def prepare_data_loader(
    data_glob,
    batch_size=TC["batch_size"],
    epoch_size=TC["epoch_size"],
    crop_size=TC["crop_size"],
):
    img_paths = np.array(glob(data_glob))
    dataset = ImageDataset(img_paths)
    # Rounds the epoch size up to the nearest batch
    epoch_size = ((epoch_size + batch_size - 1) // batch_size) * batch_size
    steps_per_epoch = epoch_size // batch_size

    # Construct the loader
    datagen = dataset.flow(steps_per_epoch, batch_size, shuffle=True)
    datagen = RandomCropper(crop_size)(datagen)
    datagen = CrappifyTransformer()(datagen)
    datagen = ImageNetNormalizer()(datagen)

    return datagen
