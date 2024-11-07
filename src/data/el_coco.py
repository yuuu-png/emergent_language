import random
import json
from typing import Optional, List

import pytorch_lightning as L
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
from transformers import GPT2TokenizerFast
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.augmentation import GaussianBlur, SimCLRDataTransform

from torch.nn.utils.rnn import pad_sequence


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


class ELNLDataset(data.Dataset):
    def __init__(self, table: str):
        with open(table, "r") as f:
            self.table = json.load(f)

    def __len__(self):
        return len(self.table["data"])

    def __getitem__(self, idx):
        return self.table["data"][idx], self.table["data"][idx]


class EmergentLanguageAndCaptionDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        wandb_logger: WandbLogger,
        val_rate: float = 0.2,
        train_artifact_path="ishiyama-k/simclr_dataset/run-yasrkseo-train:v0",
        val_artifact_path="ishiyama-k/simclr_dataset/run-yasrkseo-val:v0",
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_rate = val_rate
        self.wandb_logger = wandb_logger
        self.train_artifact_path = train_artifact_path
        self.val_artifact_path = val_artifact_path

        self.nl_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        self.nl_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.nl_transform = sequential_transforms(
            self.nl_tokenizer.encode, self.nl_tensor_transform
        )

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int], BOS_IDX, EOS_IDX):
        return torch.cat(
            (
                torch.tensor([BOS_IDX], dtype=torch.long),
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor([EOS_IDX], dtype=torch.long),
            )
        )

    def nl_tensor_transform(self, token_ids: List[int]):
        return self.tensor_transform(
            token_ids, self.nl_tokenizer.bos_token_id, self.nl_tokenizer.eos_token_id
        )

    def prepare_data(self) -> None:
        self.train_artifact = wandb.Api().artifact(self.train_artifact_path)
        self.val_table = wandb.Api().artifact(self.val_artifact_path)
        self.train_table_path = self.train_artifact.download()
        self.val_table_path = self.val_table.download()

    @property
    def nl_vocab_size(self):
        return self.nl_tokenizer.vocab_size

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = ELNLDataset(f"{self.train_table_path}/train.table.json")
            train_set_size = int(len(full_dataset) * (1 - self.val_rate))
            valid_set_size = len(full_dataset) - train_set_size

            self.train_dataset, self.val_dataset = data.random_split(
                full_dataset,
                [train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(42),
            )

        elif stage == "test":
            self.test_dataset = ELNLDataset(f"{self.val_table_path}/val.table.json")

    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for sample, _ in batch:
            # ここでsrcとtgtとの順番を合わせている
            tgt_sample, src_sample = sample
            tgt_sample = random.choice(tgt_sample)
            src_batch.append(torch.tensor(src_sample))
            tgt_batch.append(self.nl_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(
            tgt_batch, padding_value=self.nl_tokenizer.pad_token_id
        )
        return src_batch, tgt_batch

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
