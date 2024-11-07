import random
from typing import Optional, Tuple, Any

import pytorch_lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms


from src.data.augmentation import (
    simCLROriginalDataaugmentation,
    SimCLRDataTransform,
)
from src.data.cifar10 import cifar10DataAugmentation


class CocoDatasetInstance(data.Dataset):
    def __init__(
        self,
        root="/data/coco/train2017",
        annFile="/data/coco/annotations/instances_train2017.json",
        base_transform: Optional[callable] = None,
        simclr_transform: Optional[callable] = None,
        larger_transform: Optional[callable] = None,
        smaller_transform: Optional[callable] = None,
        smallest_smaller: Optional[int] = None,
        fix_smallers_num: Optional[int] = None,
        max_smallers_num: Optional[int] = None,
    ):
        self.cocoDetaction = datasets.CocoDetection(
            root, annFile, transform=base_transform
        )
        self.simclr_transform = simclr_transform
        self.larger_transform = larger_transform
        self.smaller_transfrom = smaller_transform
        self.smallest_smaller = smallest_smaller
        self.fix_smallers_num = fix_smallers_num
        self.max_smallers_num = max_smallers_num

    def __len__(self):
        return self.cocoDetaction.__len__()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.cocoDetaction[index]
        if self.simclr_transform is not None:
            simclr = self.simclr_transform(image)
        else:
            simclr = (image, image, image)
        if self.larger_transform is not None:
            larger = self.larger_transform(image)
        else:
            larger = image

        random_target = random.sample(target, len(target))
        smallers = []
        i = 0
        for annotation in random_target:
            x, y, w, h = annotation["bbox"]
            if self.smallest_smaller is not None and w * h < self.smallest_smaller:
                continue
            smaller_original = image.crop((x, y, x + w, y + h))
            if self.smaller_transfrom is not None:
                smaller = self.smaller_transfrom(smaller_original)
            else:
                smaller = smaller_original
            smallers.append(smaller)

            i += 1
            if self.max_smallers_num is not None and i >= self.max_smallers_num:
                break

        return simclr, larger, smallers, target


class CocoDetectionDataModule(L.LightningDataModule):
    """3*32*32 images from CIFAR10 dataset."""

    def __init__(
        self,
        data_dir: str = "/data/coco",
        num_workers: int = 4,
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        val_rate: float = 0.2,
        size: Optional[int] = None,
        other_transforms: list = [],
        smallest_smaller: Optional[int] = None,
        max_smallers_num: Optional[int] = None,
        larger_scale=0.2,
        smaller_scale=0.2,
    ):
        super().__init__()
        size = size if size is not None else 224

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.val_rate = val_rate
        self.smallest_smaller = smallest_smaller
        self.max_smallers_num = max_smallers_num

        self.larger_transform = transforms.Compose(
            simCLROriginalDataaugmentation(crop_scale=(larger_scale, 1))
            + other_transforms
            + [transforms.ToTensor()]
        )
        transform = transforms.Compose(
            simCLROriginalDataaugmentation()
            + other_transforms
            + [transforms.ToTensor()]
        )
        self.simclr_transform = SimCLRDataTransform(
            return_original_image=True,
            resize_original=True,
            transform1=transform,
            transform2=transform,
        )
        self.smaller_transform = transforms.Compose(
            cifar10DataAugmentation(size=size, scale=smaller_scale)
            + other_transforms
            + [transforms.ToTensor()]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full = CocoDatasetInstance(
                root=f"{self.data_dir}/train2017",
                annFile=f"{self.data_dir}/annotations/instances_train2017.json",
                simclr_transform=self.simclr_transform,
                larger_transform=self.larger_transform,
                smaller_transform=self.smaller_transform,
                smallest_smaller=self.smallest_smaller,
                max_smallers_num=self.max_smallers_num,
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
            self.test_set = CocoDatasetInstance(
                root=f"{self.data_dir}/val2017",
                annFile=f"{self.data_dir}/annotations/instances_val2017.json",
                simclr_transform=self.simclr_transform,
                larger_transform=self.larger_transform,
                smaller_transform=self.smaller_transform,
                smallest_smaller=self.smallest_smaller,
                max_smallers_num=self.max_smallers_num,
            )

    def custom_collate_fn(self, batch):
        # defaltly Dataloader use torch.stack, so all of data should be the same size.
        # however this dataset contains different sizes of annotation.
        # so this custom function doesn't use torch.stack for target

        simclrs, largers, smallers_list, targets = list(zip(*batch))

        simclr_transformeds1, simclr_transformeds2, simclr_originals = list(
            zip(*simclrs)
        )
        simclr_transformeds1 = torch.stack(simclr_transformeds1)
        simclr_transformeds2 = torch.stack(simclr_transformeds2)
        simclr_originals = list(simclr_originals)
        simclrs = (simclr_transformeds1, simclr_transformeds2, simclr_originals)

        largers = torch.stack(largers)

        smallers_len_list = [len(smallers) for smallers in smallers_list]
        all_smallers = torch.stack(
            [smaller for smallers in smallers_list for smaller in smallers]
        )
        smallers = (smallers_len_list, all_smallers)

        return simclrs, largers, smallers, targets

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


if __name__ == "__main__":
    dm = CocoDetectionDataModule()
    print("setup:")
    dm.setup("fit")

    print("train_dataloader:")
    for batch in dm.train_dataloader():
        simclr, larger, smallers, y = batch
        simclr_transformed1, simclr_transformed2, simclr_original = simclr
        smallers_len, all_smallers = smallers
        print(f"simclr size: {simclr[0].shape}")
        print(f"larger size: {larger.shape}")
        print(f"num of smallers: {all_smallers.shape}")
        print(f"type of y: {type(y)}")
        break
