import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import wandb

from egg.core import find_lengths
from egg.core.baselines import MeanBaseline

from train import SimCLRModule
from src.data import (
    MNISTDataModule,
    CIFAR10DataModule,
    mnistClassLabels,
    cifar10ClassLabels,
    CocoImageOnlyDataModule,
    SuperCLEVRImageOnlyDataModule,
)
from src.models.backbone.load import get_backbone
from src.models.backbone.parser import DINO_wrapper
from src.models import (
    backbone_parser,
    Backbone,
    RnnSpeakerListener,
    TransformerSpeakerListener,
    SpeakerOutputs,
)
from src.utils import load_artifact, SaveCrossAttentionMapMixin, heatmap


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
        max_epochs=config.epoch,
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
    if dataset == "mnist":
        dm = MNISTDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            augmentation=config.data_augmentation,
        )
        class_labels = mnistClassLabels()
    elif dataset == "cifar10":
        dm = CIFAR10DataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            augmentation=config.data_augmentation,
            augmentation_min_scale=config.augmentation_min_scale,
        )
        class_labels = cifar10ClassLabels()
    elif dataset == "coco":
        dm = CocoImageOnlyDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            min_crop=1,
        )
    elif dataset == "superCLEVR":
        dm = SuperCLEVRImageOnlyDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            min_crop=1,
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

    columns = ["label", "original", "message"]
    columns += [f"{i}_heatmap" for i in range(simclrModule.max_len)]
    columns += [f"{i}" for i in range(simclrModule.max_len)]
    table = wandb.Table(columns=columns)

    module = SimCLRAttentionModule(simclrModule, table)
    trainer.test(module, datamodule=dm)

    print("uploading table...")
    wandb_logger.experiment.log({"samples": table})
    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


class SimCLRAttentionModule(L.LightningModule, SaveCrossAttentionMapMixin):
    def __init__(
        self,
        simclrModule: SimCLRModule,
        table: wandb.Table,
        images_per_batch: int = 100,
    ):
        super(SimCLRAttentionModule, self).__init__()
        self.model = simclrModule.model
        self.table = table
        self.images_per_batch = images_per_batch

        self.dataset = simclrModule.dataset
        self.image_size = simclrModule.image_size
        self.max_len = simclrModule.max_len
        if self.dataset == "coco" or self.dataset == "superCLEVR":
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

        self.set_hook(self.model.speaker.transformer)

    def training_step(self, batch, batch_idx):
        return torch.zeros(1)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, log_header="test", save_samples=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": self.backbone_lr},
                {"params": self.model.speaker.parameters(), "lr": self.speaker_lr},
                {"params": self.model.listener.parameters(), "lr": self.listener_lr},
                {
                    "params": self.model.listener_agent.parameters(),
                    "lr": self.classifier_lr,
                },
            ]
        )

    def forward(self, x):
        return self.model(x)

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_samples=False,
    ):
        if self.dataset == "superCLEVR":
            x = batch
            y = torch.zeros(len(x), device=x.device)
        else:
            x, y = batch

        n = x.size(0)

        self.reset_attention_map()
        listener_outputs, speaker_outputs = self.model(x)
        attention_map = self.attention_map

        if self.classifier:
            z, class_logits = listener_outputs
        else:
            z = listener_outputs

        message_length = find_lengths(speaker_outputs.sequence)

        if not save_samples:
            return

        channel, height, width = self.image_size

        if self.dataset == "coco" or self.dataset == "superCLEVR":
            labels = [0 for _ in range(n)]
        else:
            labels = y.cpu().numpy()
        originals = x.view(n, channel, height, width).cpu().numpy()
        message = speaker_outputs.sequence.cpu().numpy()

        attention_map = attention_map[-1]

        image_num, seq_len, pixel_num = attention_map.size()

        fixed_attention_map = torch.zeros_like(attention_map)
        for i in range(image_num):
            fixed_attention_map[i] = (
                attention_map[i] - attention_map[i].sum(dim=0, keepdim=True) / seq_len
            )

        fixed_attention_map = F.relu(fixed_attention_map)

        patch_size = int(math.sqrt(pixel_num))
        atten_map_seq = torch.zeros(n, self.max_len, channel, height, width)
        # 並列化したかったが，あきらめた
        for l in range(self.max_len):
            atten_map_seq[:, l] = heatmap(
                x,
                fixed_attention_map[:, l].view(-1, patch_size, patch_size),
                normalize="linear",
                heatmap_conc=0.6,
            )
        atten_map_seq = atten_map_seq.cpu().numpy()

        _id = 0
        for lbl, org, msg, map in zip(labels, originals, message, atten_map_seq):
            if channel == 1:
                org = wandb.Image(org)
            else:
                org = wandb.Image(org.transpose(1, 2, 0))
            map_ = []
            for i in range(self.max_len):
                map_.append(wandb.Image(map[i].transpose(1, 2, 0)))
            self.table.add_data(self.labels[lbl], org, msg, *map_, *msg[:-1])
            _id += 1
            if _id >= self.images_per_batch:
                break
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
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--limit_batches", type=float, default=None)

    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--speaker_lr", type=float, default=2e-5)
    parser.add_argument("--listener_lr", type=float, default=1e-4)
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)

    # Game parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "cifar10", "coco", "superCLEVR"],
    )
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
    parser.add_argument("--project", type=str, default="simclr_attention")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--images_per_batch", type=int, default=20)
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
