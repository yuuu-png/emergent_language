import random

import torch
from PIL import ImageFilter
from torchvision import datasets, transforms


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def simCLROriginalDataaugmentation(
    size: int = 224, s: float = 1.0, crop_scale=(0.08, 1)
):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transformations = [
        transforms.RandomResizedCrop(size=size, scale=crop_scale),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),  # with 0.5 probability
    ]
    return transformations


class SimCLRDataTransform:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self,
        return_original_image: bool = False,
        resize_original: bool = False,
        original_size: int = 64,
        transform1: transforms.Compose = None,
        transform2: transforms.Compose = None,
    ):

        self.transform1 = transform1
        self.transform2 = transform2

        self.return_original_image = return_original_image
        if self.return_original_image:
            self.original_image_transform = transforms.Compose(
                ([transforms.Resize(size=original_size)] if resize_original else [])
                + [transforms.ToTensor()]
            )

    def __call__(self, x):
        x_i = self.transform1(x)
        x_j = self.transform2(x)
        if self.return_original_image:
            return x_i, x_j, self.original_image_transform(x)
        return x_i, x_j
