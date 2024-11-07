from typing import Optional
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import pytorch_lightning as L
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from src.data.augmentation import (
    simCLROriginalDataaugmentation,
    SimCLRDataTransform,
)


class SuperCLEVRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = [
            f"superCLEVR_new_{i:06d}.png" for i in range(30000)
        ]  # 000000~299999までのファイル名を生成

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")  # 画像をRGB形式で読み込み
        if self.transform:
            image = self.transform(image)
        return image


class SuperCLEVRImageOnlyDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/superCLEVR",
        num_workers: int = 4,
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        val_rate_to_train_and_val: float = 0.2,
        test_rate_to_full: float = 0.2,
        size: Optional[int] = None,
        min_crop: float = 1,
    ):
        super().__init__()
        size = size if size is not None else 224

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_rate_to_train_and_val = val_rate_to_train_and_val
        self.test_rate_to_full = test_rate_to_full
        self.test_set = None

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(min_crop, 1.0)),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "test" and self.test_set is None:
            full = SuperCLEVRDataset(
                data_dir=self.data_dir,
                transform=self.transform,
            )

            train_val_set_size = int(len(full) * (1 - self.val_rate_to_train_and_val))
            train_set_size = int(
                train_val_set_size * (1 - self.val_rate_to_train_and_val)
            )
            valid_set_size = train_val_set_size - train_set_size
            test_set_size = len(full) - train_set_size - valid_set_size

            self.train_set, self.val_set, self.test_set = data.random_split(
                full,
                [train_set_size, valid_set_size, test_set_size],
                generator=torch.Generator().manual_seed(42),
            )

    # def custom_collate_fn(self, batch):
    #     images = zip(*batch)
    #     images = torch.stack(images)
    #     return images

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=self.custom_collate_fn,
        )
