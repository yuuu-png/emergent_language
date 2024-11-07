from typing import Optional

import pytorch_lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms


from src.data.augmentation import (
    simCLROriginalDataaugmentation,
    SimCLRDataTransform,
)


class CocoCaptionsDataModule(L.LightningDataModule):
    """3*32*32 images from CIFAR10 dataset.
    if you set sub_min_crop, it will use two different crop scale for simclr.
    otherwise, it will use same crop scale for simclr.
    """

    def __init__(
        self,
        data_dir: str = "/data/coco",
        num_workers: int = 4,
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        val_rate: float = 0.2,
        size: Optional[int] = None,
        simclr: bool = False,
        augmentation: bool = False,
        other_transforms: list = [],
        min_crop: float = 0.08,
        sub_min_crop: float | None = None,
        sub_max_crop: float | None = None,
    ):
        super().__init__()
        size = size if size is not None else 224

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_rate = val_rate
        self.augmentation = augmentation
        self.simclr = simclr
        self.augmentation_transform = transforms.Compose(
            simCLROriginalDataaugmentation(crop_scale=(min_crop, 1.0))
        )

        if simclr:

            def transform(min_crop: float, max_crop: float = 1.0):
                return transforms.Compose(
                    simCLROriginalDataaugmentation(crop_scale=(min_crop, max_crop))
                    + other_transforms
                    + [transforms.ToTensor()],
                )

            self.transform = SimCLRDataTransform(
                return_original_image=True,
                resize_original=True,
                transform1=transform(min_crop),
                transform2=transform(
                    sub_min_crop if sub_min_crop else min_crop,
                    sub_max_crop if sub_max_crop else 1.0,
                ),
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
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full = datasets.CocoCaptions(
                root=f"{self.data_dir}/train2017",
                annFile=f"{self.data_dir}/annotations/captions_train2017.json",
                transform=self.transform,
            )

            train_set_size = int(len(full) * (1 - self.val_rate))
            valid_set_size = len(full) - train_set_size

            self.train_set, self.val_set = data.random_split(
                full,
                [train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = datasets.CocoCaptions(
                root=f"{self.data_dir}/val2017",
                annFile=f"{self.data_dir}/annotations/captions_val2017.json",
                transform=self.transform,
            )

    def custom_collate_fn(self, batch):
        # defaltly Dataloader use torch.stack, so all of data should be the same size.
        # however this dataset contains different sizes of annotation.
        # so this custom function doesn't use torch.stack for target
        images1 = []
        images2 = []
        original_images = []
        targets = []
        for sample in batch:
            image, target = sample
            i1, i2, o = image
            images1.append(i1)
            images2.append(i2)
            original_images.append(o)
            targets.append(target)
        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        # original_images = torch.stack(original_images)
        images = images1, images2, original_images
        return images, targets

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )
