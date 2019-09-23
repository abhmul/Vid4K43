from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor
from torchvision.transforms.functional import resize

from PIL import Image

import os
from glob import glob
import sys

identity = lambda x: x


class ImagePathDataset(VisionDataset):
    def __init__(self, paths, transform=identity, loader=default_loader):
        """A dataset of image paths"""
        assert len(paths)
        root = os.path.dirname(paths[0])
        super().__init__(root, transform=transform, target_transform=None)
        self.paths = paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.paths[index]
        return {"input": self.transform(self.loader(path))}

    def __len__(self):
        return len(self.paths)


def Crappify(factor):
    def crappify(img):
        h, w = img.size
        img = resize(img, (round(h * factor), round(w * factor)))
        img = resize(img, (h, w))
        return img

    return crappify


default_crappifer = Crappify(0.5)


class CrappifyDataset(ImagePathDataset):
    def __init__(
        self,
        glob_path,
        crappifier=default_crappifer,
        transform=identity,
        crop_size=None,
        loader=default_loader,
        test=False,
    ):
        """Loads a dataset for decrappficiation
        
        Arguments:
            glob_path {string} -- Path that describes glob for all images in dataset
        
        Keyword Arguments:
            crappifier {function} -- a crappficiation function for the input image
            transform {function} -- a transform function for the image before crappficiation.
                Transforms both input and target images with the same transform
            loader {function} -- function to load image given path (default: {default_loader})
        """
        self.cropper = None
        if crop_size is not None:
            self.cropper = RandomCrop(crop_size)
            transform = Compose([self.cropper, transform])
        super().__init__(sorted(glob(glob_path)), transform=transform, loader=loader)
        self.glob_path = glob_path
        self.crappifier = crappifier
        self.test = test
        toTensor = ToTensor()
        self.tensorify = lambda x: toTensor(x)

    @property
    def crop_size(self):
        if self.cropper is None:
            raise ValueError("This dataset is not cropping")
        return self.cropper.size

    @crop_size.setter
    def crop_size(self, size):
        if self.cropper is None:
            raise ValueError("This dataset is not cropping")
        self.cropper.size = size

    def __getitem__(self, index):
        sample = super().__getitem__(index)["input"]
        x = sample
        x = self.crappifier(x)
        x = self.tensorify(x)
        if self.test:
            return {"input": x}
        return {"input": x, "target": self.tensorify(y)}}


class CritDataset(ImagePathDataset):
    def __init__(
        self,
        generated_glob_path,
        original_glob_path,
        transform=identity,
        crop_size=None,
        loader=default_loader,
    ):
        self.cropper = None
        if crop_size is not None:
            self.cropper = RandomCrop(crop_size)
            transform = Compose([self.cropper, transform])
        generated_paths = sorted(glob())
        original_paths = sorted(glob(original_glob_path))
        super().__init__(
            generated_paths + original_paths, transform=transform, loader=loader
        )

        self.generated_glob_path = generated_glob_path
        self.generated_paths = generated_paths
        self.num_generated = len(generated_paths)
        self.original_glob_path = original_glob_path
        self.original_paths = original_paths
        self.num_original = len(original_paths)

        toTensor = ToTensor()
        self.tensorify = lambda x: toTensor(x)

    @property
    def crop_size(self):
        if self.cropper is None:
            raise ValueError("This dataset is not cropping")
        return self.cropper.size

    @crop_size.setter
    def crop_size(self, size):
        if self.cropper is None:
            raise ValueError("This dataset is not cropping")
        self.cropper.size = size

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        x = self.tensorify(sample)
        y = torch.tensor(0.0 if index < self.num_generated else 1.0)
        return {"input": x, "target": y}
