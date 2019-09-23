import numpy as np
import torchvision.transforms.functional as F
import pyjet.data as data
import pyjet.augmenters as aug
import pyjet.backend as J
import logging
from scipy.misc import imresize
from skimage.util import random_noise

"""This file contains the various data objects and manipulators
I'll use to handle the data. The RandomCropper implements a pyjet
augmenter and crops the given images to 
"""


class DataTransformer(object):
    """This class is designed to transform a stream
    of data into some new output data. More specifically
    it takes the output of a DatasetGenerator and yields
    an output of its own
    """

    def __init__(self, labels=False):
        self.labels = labels

    def transform(self, batch):
        raise NotImplementedError

    def __call__(self, generator):
        return TransformerGenerator(self, generator)


class TransformerGenerator(data.BatchGenerator):
    def __init__(self, transformer, generator):
        # Copy the steps per epoch and batch size if it has one
        if hasattr(generator, "steps_per_epoch") and hasattr(generator, "batch_size"):
            super(TransformerGenerator, self).__init__(
                steps_per_epoch=generator.steps_per_epoch,
                batch_size=generator.batch_size,
            )
        else:
            logging.warning(
                "Input generator does not have a "
                "steps_per_epoch or batch_size "
                "attribute. Continuing without them."
            )

        self.transformer = transformer
        self.generator = generator

    def __next__(self):
        return self.transformer.transform(next(self.generator))


def batch_apply(func, x):
    x_out = np.stack([func(xi) for xi in x], axis=0)
    return x_out


def crappfier(factor, noise=True):
    """A simple function to get a bilinear resizer with given factor"""

    # Do nothing if we're not crappifying
    if factor == 1.0:
        return lambda x: x

    def crappify_img(npimg):
        npimg = imresize(
            imresize(npimg, factor, interp="bilinear"), 1 / factor, interp="bilinear"
        )
        if noise:
            npimg = random_noise(npimg)
        else:
            # convert to float
            npimg = npimg / 255.0

        return npimg

    return lambda x: batch_apply(crappify_img, x)


def cropper(crop_size):
    def crop_img(npimg):
        h, w, _ = npimg.shape
        ch, cw = crop_size
        # If the image is too small, we'll make it bigger
        # This copies the entire image which might not be necessary
        # If speed becomes and issue look here.
        if h < ch or w < cw:
            newimg_shape = (min(ch, h), min(cw, w))
            new_npimg = np.zeros(newimg_shape + npimg.shape[2:])
            new_npimg[[slice(i) for i in npimg.shape]] = npimg
            npimg = new_npimg
            h, w, _ = npimg.shape

        cropy = np.random.randint(h - ch + 1)
        cropx = np.random.randint(w - cw + 1)
        return npimg[cropy : cropy + ch, cropx : cropx + cw]

    return lambda x: batch_apply(crop_img, x)


def channels_first(x):
    return x.transpose(0, 3, 1, 2).astype(np.float32)


def transform(x, crop_size, factor, noise=True):
    crap = crappfier(factor, noise=noise)
    crop = cropper(crop_size)
    # Crop the image and crappify
    x = crop(x)
    y = x / 255.0  # get the label and convert to float
    x = crap(x)
    # Convert to channels first
    return channels_first(x), channels_first(y)


class PrepareData(DataTransformer):
    def __init__(self, crop_size, factor, noise):
        super().__init__(labels=False)
        self.crop_size = crop_size
        self.factor = factor
        self.noise = noise

    def transform(self, x):
        return transform(x, self.crop_size, self.factor, noise=self.noise)
