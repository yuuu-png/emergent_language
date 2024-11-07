from typing import Optional

import pytorch_lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms


class CocoImageOnlyDataModule(L.LightningDataModule):
    """3*32*32 images from CIFAR10 dataset."""

    def __init__(
        self,
        data_dir: str = "/data/coco",
        num_workers: int = 4,
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        val_rate: float = 0.2,
        size: Optional[int] = None,
        min_crop: float = 1,
    ):
        super().__init__()
        size = size if size is not None else 224

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_rate = val_rate
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(min_crop, 1.0)),
                transforms.ToTensor(),
            ],
        )

    # def on_after_batch_transfer(
    #     self, batch: torch.Any, dataloader_idx: int
    # ) -> torch.Any:
    #     x, y = batch
    #     x = torch.stack([self.transform(img) for img in x])
    #     return x, y

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
        images, targets = zip(*batch)
        images = torch.stack(images)
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
