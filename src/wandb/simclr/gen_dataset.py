import argparse
from collections import defaultdict
import os
import sys

import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy
import wandb
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data

from egg.core import find_lengths

from train import SimCLRModule
from src.models import backbone_parser
from src.utils import load_artifact


class CocoDataTransform:
    def __init__(self, size, mini_size):
        self.size = size
        self.mini_size = mini_size

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(1, 1)),
                transforms.ToTensor(),
            ]
        )
        self.mini_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=mini_size, scale=(1, 1)),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        larger = self.transform(x)
        mini = self.mini_transform(x)
        return larger, mini


class CocoCaptionsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/coco",
        num_workers: int = 4,
        train_batch_size: int = 64,
        size: int = 224,
        mini_size=32,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.transform = CocoDataTransform(size, mini_size)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_set = datasets.CocoCaptions(
                root=f"{self.data_dir}/train2017",
                annFile=f"{self.data_dir}/annotations/captions_train2017.json",
                transform=self.transform,
            )
            self.val_set = datasets.CocoCaptions(
                root=f"{self.data_dir}/val2017",
                annFile=f"{self.data_dir}/annotations/captions_val2017.json",
                transform=self.transform,
            )

    def custom_collate_fn(self, batch):
        # defaltly Dataloader use torch.stack, so all of data should be the same size.
        # however this dataset contains different sizes of annotation.
        # so this custom function doesn't use torch.stack for target
        originals = []
        minis = []
        targets = []
        for sample in batch:
            image, target = sample
            original, mini = image
            originals.append(original)
            minis.append(mini)
            targets.append(target)
        originals = torch.stack(originals)
        minis = torch.stack(minis)
        images = originals, minis
        return images, targets

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )


def main(config, sweep=False):
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    offline = config.offline

    args = [arg.lstrip("-") for arg in sys.argv[1:] if len(arg) < 64]
    if config.name is not None:
        name = config.name
    elif len(args) > 0:
        name = ",".join(args[:2])
    else:
        name = None
    wandb_logger = WandbLogger(
        name=name,
        project=config.project,
        save_dir="/work/wandb-logs",
        log_model=not (sweep or offline),
        config=config,
        mode="disabled" if offline else "online",
        tags=args,
    )

    # training settings
    trainer = L.Trainer(
        devices="auto" if config.gpu is None else config.gpu,
        max_epochs=1,
        logger=wandb_logger,
        fast_dev_run=config.fast_dev_run,
        deterministic=True,
        callbacks=[
            ModelSummary(max_depth=2),
        ],
        limit_predict_batches=config.limit_batches,
        limit_test_batches=config.limit_batches,
        limit_train_batches=config.limit_batches,
        limit_val_batches=config.limit_batches,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if config.gpu is None
            and torch.cuda.device_count() > 1
            or config.gpu is not None
            and len(config.gpu) > 1
            else "auto"
        ),
    )

    # load the model
    checkpoint = load_artifact(
        artifact_id=config.checkpoint,
        wandb_logger=wandb_logger,
        global_rank=trainer.global_rank,
    )
    simclrModule = SimCLRModule.load_from_checkpoint(
        checkpoint,
        map_location=lambda storage, loc: storage,
        backbone_checkpoint=None,
        backbone_dim=config.backbone_dim,
    )
    simclrModule: SimCLRModule = torch.compile(simclrModule, mode="default")

    # load the data
    num_workers = os.cpu_count() if config.num_workers is None else config.num_workers
    num_workers = 32 if num_workers > 32 else num_workers
    dataset = (
        config.dataset
        if config.dataset is not None
        else simclrModule.hparams["dataset"]
    )
    if dataset == "coco":
        dm = CocoCaptionsDataModule(
            train_batch_size=config.batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

    trainer = L.Trainer(
        devices=1,
        num_nodes=1,
        logger=wandb_logger,
        deterministic=True,
        fast_dev_run=config.fast_dev_run,
        limit_predict_batches=config.limit_batches,
        limit_test_batches=config.limit_batches,
        limit_train_batches=config.limit_batches,
        limit_val_batches=config.limit_batches,
    )

    columns = ["captions", "el"]
    train = wandb.Table(columns=columns)
    val = wandb.Table(columns=columns)

    module = SimCLRGenerateDataset(simclrModule, train, val)
    trainer.fit(module, datamodule=dm)

    print("uploading table...")
    wandb_logger.experiment.log({"train": train, "val": val})
    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


class SimCLRGenerateDataset(L.LightningModule):
    def __init__(
        self,
        simclrModule: SimCLRModule,
        train: wandb.Table,
        val: wandb.Table,
    ):
        super(SimCLRGenerateDataset, self).__init__()
        self.model = simclrModule.model
        for param in self.model.parameters():
            param.requires_grad = False

        self.train_table = train
        self.val_table = val

        self.dataset = simclrModule.dataset
        self.image_size = simclrModule.image_size
        self.max_len = simclrModule.max_len
        if self.dataset == "coco":
            self.labels = defaultdict(lambda: "None")
        else:
            self.labels = simclrModule.labels
        self.backbone_lr = simclrModule.backbone_lr
        self.speaker_lr = simclrModule.speaker_lr
        self.listener_lr = simclrModule.listener_lr
        self.classifier_lr = simclrModule.classifier_lr
        self.lazy_speaker_beta1 = simclrModule.lazy_speaker_beta1
        self.lazy_speaker_beta2 = simclrModule.lazy_speaker_beta2
        self.classifier = simclrModule.classifier

    def training_step(self, batch, batch_idx):
        self._step(batch, batch_idx, log_header="train", table=self.train_table)
        return torch.zeros(1, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log_header="val", table=self.val_table)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def forward(self, x):
        return self.model(x)

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        table: wandb.Table,
    ):
        x, y = batch
        original, mini_image = x

        n = original.size(0)

        listener_outputs, speaker_outputs = self.model(original)

        if self.classifier:
            z, class_logits = listener_outputs
        else:
            z = listener_outputs

        message_length = find_lengths(speaker_outputs.sequence)

        channel, height, width = self.image_size

        message = speaker_outputs.sequence.cpu().numpy()
        mini_image = mini_image.cpu().numpy()

        _id = 0
        for msg, cap in zip(message, y):
            table.add_data(cap, msg)
            _id += 1
        return

    def _lazy_speaker_alpha(self, reconst_loss: torch.Tensor):
        """adaptive regularization coeffient for lazy speaker
        This is explained in the paper “LazImpa” A.1.4
        This program is self-supervised learning, so we cannot access the accuracy.
        So we use the reconstruction loss instead of the accuracy.
        """
        preudo_acc = 1 - reconst_loss
        return torch.pow(preudo_acc, self.lazy_speaker_beta1) / self.lazy_speaker_beta2


def create_parser():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument("--gpu", nargs="*", default=None, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--limit_batches", type=float, default=None)

    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--speaker_lr", type=float, default=2e-5)
    parser.add_argument("--listener_lr", type=float, default=1e-4)
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)

    # Game parameters
    parser.add_argument("--dataset", type=str, default=None, choices=["coco"])
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=8)
    parser.add_argument("--length_cost", type=float, default=0.0)
    # SimCLR parameters
    parser.add_argument(
        "--similarity", type=str, default="cosine", choices=["cosine", "dot"]
    )
    parser.add_argument("--cosine_temperature", type=float, default=0.1)

    # Model parameters
    parser = backbone_parser(parser)
    parser.add_argument(
        "--arch", type=str, default="transformer", choices=["rnn", "transformer"]
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="hidden dimension for RNN and d_model for Transformer",
    )
    # RNN parameters
    parser.add_argument("--embed_dim", type=int, default=10)
    # Transformer parameters
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Lazimpa
    parser.add_argument("--lazy_speaker", action="store_true")
    parser.add_argument("--lazy_speaker_beta1", type=float, default=45)
    parser.add_argument("--lazy_speaker_beta2", type=float, default=10)
    # Classifier parameters
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_hidden_dim", type=int, default=64)
    parser.add_argument("--cls_loss_coeff", type=float, default=1)

    # Optimization parameters
    parser.add_argument("--gumbel_softmax", action="store_true")
    parser.add_argument("--straight_through", action="store_true")
    parser.add_argument("--detach_message", action="store_true")
    # REINFORCE algorithm parameters
    parser.add_argument("--wo_policy_loss", action="store_true")
    parser.add_argument("--entropy_coeff", type=float, default=0.010)

    # model settings
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--backbone_dim",
        type=int,
        default=None,
        help="this property is used when load a cnn backbone but need to specify its dim becouse this code can't tell it.",
    )

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="simclr_dataset")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
