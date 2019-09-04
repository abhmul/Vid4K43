import numpy as np
import torchvision.transforms.functional as F
import pyjet.data as data
import pyjet.augmenters as aug
import pyjet.backend as J
import logging
from scipy.misc import imresize

"""This file contains the various data objects and manipulators
I'll use to handle the data. The RandomCropper implements a pyjet
augmenter and crops the given images to 
"""


class RandomCropper(aug.Augmenter):
    def __init__(self, crop_size, labels=False, augment_labels=False):
        super(RandomCropper, self).__init__(
            labels=labels, augment_labels=augment_labels
        )
        self.crop_size = crop_size

    @property
    def crop_width(self):
        self.crop_size[1]

    @property
    def crop_height(self):
        self.crop_size[0]

    def augment(self, x):
        """The input comes in as an np array of
        np arrays that make up the individual images.
        We randomly crop them to the specified size.
        """

        def crop_img(npimg):
            h, w, _ = npimg.shape
            # If the image is too small, we'll make it bigger
            # This copies the entire image which might not be necessary
            # If speed becomes and issue look here.
            if h < self.crop_height or w < self.crop_width:
                newimg_shape = (min(self.crop_height, h), min(self.crop_width, w))
                new_npimg = np.zeros(newimg_shape + npimg.shape[2:])
                new_npimg[[slice(i) for i in npimg.shape]] = npimg
                npimg = new_npimg
                h, w, _ = npimg.shape

            cropy = np.random.randint(h - self.crop_height + 1)
            cropx = np.random.randint(w - self.crop_width + 1)
            return npimg[
                cropy : cropy + self.crop_height, cropx : cropx + self.crop_width
            ]

        x = [crop_img(npimg) for npimg in x]
        assert all(x[0].shape == x[i].shape for i in range(1, len(x)))
        x = np.array(x)
        return x


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


# Goes after crappifier
class ImageNetNormalizer(DataTransformer):
    def __init__(self):
        super().__init__(labels=True)
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.eps = J.epsilon

    def normalize_uint8(self, x):
        assert x.max() > 200
        x = np.clip(x, 0, 255)
        x = x / 255.0
        return (x - self.mean) / (self.std + self.eps)

    def transform(self, batch):
        x, y = batch
        # x is batched (B x 3 x H x W) and is uint8
        x = self.normalize_uint8(x)
        y = self.normalize_uint8(y)
        return x, y


def bilinear(factor):
    """A simple function to get a bilinear resizer with given factor"""
    return lambda x: imresize(
        imresize(x, factor, interp="bilinear"), 1 / factor, interp="bilinear"
    )


class CrappifyTransformer(DataTransformer):
    def __init__(self, crappifier=bilinear(0.5)):
        super(CrappifyTransformer, self).__init__(labels=False)
        self.crappifier = crappifier

    def transform(self, x):
        return self.crappifier(x), x
