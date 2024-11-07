from typing import Optional
import pytorch_lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

from src.data.augmentation import GaussianBlur, SimCLRDataTransform


def cifar10DataAugmentation(size: int = 32, s: float = 0.5, scale: float = 0.08):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transformations = [
        transforms.RandomResizedCrop(size=size, antialias=True, scale=(scale, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),  # with 0.5 probability
    ]
    return transformations


def cifar10ClassLabels():
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


class CIFAR10DataModule(L.LightningDataModule):
    """3*32*32 images from CIFAR10 dataset."""

    def __init__(
        self,
        data_dir: str = "/data",
        num_workers: int = 4,
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        val_rate: float = 0.2,
        size: Optional[int] = None,
        simclr: bool = False,
        augmentation: bool = False,
        augmentation_min_scale: float = 0.08,
        other_transforms: list = [],
    ):
        super().__init__()
        size = size if size is not None else 32

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_rate = val_rate
        self.augmentation = augmentation
        self.simclr = simclr
        self.augmentation_transform = transforms.Compose(
            cifar10DataAugmentation(size=size, scale=augmentation_min_scale)
        )

        if simclr:
            transform = transforms.Compose(
                cifar10DataAugmentation(size=size, scale=augmentation_min_scale)
                + other_transforms
                + [transforms.ToTensor()]
            )
            self.transform = SimCLRDataTransform(
                return_original_image=True,
                transform1=transform,
                transform2=transform,
            )
        else:
            self.transform = transforms.Compose(
                other_transforms + [transforms.ToTensor()]
            )

    def on_after_batch_transfer(
        self, batch: torch.Any, dataloader_idx: int
    ) -> torch.Any:
        if self.simclr:
            return batch
        if not (self.trainer.training and self.augmentation):
            return batch

        x, y = batch
        x = torch.stack([self.augmentation_transform(img) for img in x])
        return x, y

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )

            train_set_size = int(len(mnist_full) * (1 - self.val_rate))
            valid_set_size = len(mnist_full) - train_set_size

            self.mnist_train, self.mnist_val = data.random_split(
                mnist_full,
                [train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.mnist_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.mnist_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.mnist_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
