import argparse
import os
from pathlib import Path
import sys
from typing import Optional

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

from src.data import (
    MNISTDataModule,
    CIFAR10DataModule,
    mnistClassLabels,
    cifar10ClassLabels,
)
import src.models.generator as generators
from src.models.backbone.load import get_backbone
from src.models import (
    backbone_parser,
    RnnSpeakerListener,
    TransformerSpeakerListener,
)
from src.utils.utils import load_artifact


def main(config, sweep=False):
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    offline = config.offline

    # load dataset
    num_workers = os.cpu_count() if config.num_workers is None else config.num_workers
    num_workers = 32 if num_workers > 32 else num_workers
    if config.dataset == "mnist":
        dm = MNISTDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            augmentation=config.data_augmentation,
        )
    elif config.dataset == "cifar10":
        dm = CIFAR10DataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            augmentation=config.data_augmentation,
            augmentation_min_scale=config.augmentation_min_scale,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

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
    val_checkpoint = ModelCheckpoint(
        filename="best-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_reconstruction_loss",
        mode="min",
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_reconstruction_loss",
        mode="min",
        patience=config.early_stopping_patience,
    )
    early_stopping_entropy = EarlyStopping(
        monitor="val_entropy",
        stopping_threshold=1e-2,
        patience=config.epoch,
    )
    dm.prepare_data()
    dm.setup("fit")
    num_batches = len(dm.train_dataloader())
    trainer = L.Trainer(
        devices="auto" if config.gpu is None else config.gpu,
        max_epochs=config.epoch,
        logger=wandb_logger,
        fast_dev_run=config.fast_dev_run,
        deterministic=True,
        callbacks=[
            val_checkpoint,
            early_stopping,
            early_stopping_entropy,
            ModelSummary(max_depth=2),
        ]
        + ([latest_checkpoint] if not sweep else []),
        limit_predict_batches=config.limit_batches,
        limit_test_batches=config.limit_batches,
        limit_train_batches=config.limit_batches,
        limit_val_batches=config.limit_batches,
        log_every_n_steps=50 if num_batches > 50 else num_batches // 4,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if config.gpu is None
            and torch.cuda.device_count() > 1
            or config.gpu is not None
            and len(config.gpu) > 1
            else "auto"
        ),
    )

    backbone_checkpoint_path = (
        load_artifact(
            artifact_id=config.backbone_checkpoint,
            wandb_logger=wandb_logger,
            global_rank=trainer.global_rank,
        )
        if config.backbone_checkpoint is not None
        else None
    )

    module = GenerateModule(
        backbone_lr=config.backbone_lr,
        speaker_lr=config.speaker_lr,
        listener_lr=config.listener_lr,
        listener_agent_lr=config.generator_lr,
        max_len=config.max_len,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        entropy_coeff=config.entropy_coeff,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        length_const=config.length_cost,
        reconstruction_loss=config.reconstruction_loss,
        arch=config.arch,
        nhead=config.nhead,
        dropout=config.dropout,
        num_layers=config.num_layers,
        gumbel_softmax=config.gumbel_softmax,
        straight_through=config.straight_through,
        wo_policy_loss=config.wo_policy_loss,
        detach_message=config.detach_message,
        backbone=config.backbone,
        dataset=config.dataset,
        backbone_checkpoint=backbone_checkpoint_path,
        freeze_backbone=not config.wo_freeze,
        classifier=config.classifier,
        classifier_lr=config.classifier_lr,
        classifier_hidden_dim=config.classifier_hidden_dim,
        cls_loss_coeff=config.cls_loss_coeff,
        sweep=sweep,
    )

    if not sweep:
        wandb_logger.watch(module, log="all")

    trainer.fit(module, dm)

    # Because of DDP, there are processes of number of GPUs.
    if (config.gpu is None and torch.cuda.device_count() > 1) or (
        config.gpu is not None and len(config.gpu) > 1
    ):
        print(config.gpu)
        torch.distributed.destroy_process_group()
        if trainer.global_rank != 0:
            return

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

    if not config.fast_dev_run:
        module = GenerateModule.load_from_checkpoint(val_checkpoint.best_model_path)
    trainer.test(module, datamodule=dm)

    # save model
    if not config.fast_dev_run and not sweep:
        print("saving model...")
        module_artifact = wandb.Artifact(
            f"model-{wandb_logger.version}",
            type="model",
            metadata=vars(config),
        )
        module_artifact.add_file(val_checkpoint.best_model_path, "model.chpt")
        module_artifact.save()

    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


class GenerateModule(L.LightningModule):
    class ListenrAgent(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            class_num: int,
            dropout=0.1,
            generator: nn.Module = torch.nn.Identity(),
        ):
            super().__init__()
            self.generator = generator
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, class_num),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            image = self.generator(x)
            logits = self.classifier(x)
            return image, logits

    def __init__(
        self,
        backbone_lr: float,
        speaker_lr: float,
        listener_lr: float,
        listener_agent_lr: float,
        max_len: int,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        entropy_coeff: float,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        length_const: float,
        reconstruction_loss: str,
        arch: str,
        nhead: int,
        dropout: float,
        num_layers: int,
        gumbel_softmax: bool,
        straight_through: bool,
        wo_policy_loss: bool,
        detach_message: bool,
        backbone: str,
        dataset: str,
        freeze_backbone: bool,
        sweep: bool,
        backbone_checkpoint: Optional[Path] = None,
        backbone_dim: Optional[int] = None,
        classifier: bool = False,
        classifier_lr: float = 1e-4,
        classifier_hidden_dim: int = 64,
        cls_loss_coeff: int = 1,
    ):
        super(GenerateModule, self).__init__()
        self.save_hyperparameters()

        self.backbone_lr = backbone_lr
        self.speaker_lr = speaker_lr
        self.listener_lr = listener_lr
        self.listener_agent_lr = listener_agent_lr
        self.sweep = sweep
        self.max_len = max_len
        self.entropy_coeff = entropy_coeff
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.length_const = length_const
        self.reconstruction_loss = reconstruction_loss
        self.wo_policy_loss = wo_policy_loss
        self.classifier = classifier
        self.classifier_lr = classifier_lr
        self.cls_loss_coeff = cls_loss_coeff

        backbone = get_backbone(
            dataset=dataset,
            name=backbone,
            pretrain=backbone_checkpoint,
            freeze=freeze_backbone,
            dim=backbone_dim,
        )
        if dataset == "mnist":
            self.image_size = (1, 28, 28)
            class_num = 10
            self.labels = mnistClassLabels()
            generator = generators.Monotone28(z_dim=hidden_dim, ngf=128, nc=1)
        elif dataset == "cifar10":
            self.image_size = (3, 32, 32)
            class_num = 10
            self.labels = cifar10ClassLabels()
            generator = generators.Color32(z_dim=hidden_dim, ngf=128, nc=3)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if arch == "rnn":
            self.model = RnnSpeakerListener(
                backbone=backbone,
                max_len=max_len,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                listener_agent=self.ListenrAgent(
                    input_dim=hidden_dim,
                    hidden_dim=classifier_hidden_dim,
                    class_num=class_num,
                    generator=generator,
                ),
            )
        elif arch == "transformer":
            self.model = TransformerSpeakerListener(
                backbone=backbone,
                max_len=max_len,
                d_model=hidden_dim,
                vocab_size=vocab_size,
                nhead=nhead,
                dropout=dropout,
                num_layers=num_layers,
                gumbe_softmax=gumbel_softmax,
                straight_through=straight_through,
                detach_message=detach_message,
                listener_agent=self.ListenrAgent(
                    input_dim=hidden_dim,
                    hidden_dim=classifier_hidden_dim,
                    class_num=class_num,
                    generator=generator,
                ),
                freeze=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        self.baseline = MeanBaseline()
        self.len_baseline = MeanBaseline()

    def training_step(self, batch, batch_idx):
        save_samples = (
            batch_idx < self.batches_to_log
            and self.current_epoch % self.save_interval == 0
        )
        return self._step(
            batch,
            batch_idx,
            log_header="train",
            save_samples=save_samples,
        )

    def validation_step(self, batch, batch_idx):
        save_samples = (
            batch_idx < self.batches_to_log
            and self.current_epoch % self.save_interval == 0
        )
        self._step(
            batch,
            batch_idx,
            log_header="val",
            save_samples=save_samples,
        )

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, log_header="test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": self.backbone_lr},
                {"params": self.model.speaker.parameters(), "lr": self.speaker_lr},
                {"params": self.model.listener.parameters(), "lr": self.listener_lr},
                {
                    "params": self.model.listener_agent.parameters(),
                    "lr": self.listener_agent_lr,
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
        x, y = batch
        n = x.size(0)
        listener_outputs, speaker_outputs = self.model(x)
        x_hat, class_logits = listener_outputs
        x_hat = x_hat.reshape(n, -1)
        message_length = find_lengths(speaker_outputs.sequence)

        # standard loss
        # reconstruction_loss
        bce = F.binary_cross_entropy(x_hat, x.reshape(n, -1), reduction="none").mean(
            dim=1
        )
        mae = F.l1_loss(x_hat, x.reshape(n, -1), reduction="none").mean(dim=1)
        mse = F.mse_loss(x_hat, x.reshape(n, -1), reduction="none").mean(dim=1)
        if self.reconstruction_loss == "bce":
            reconstruction_loss = bce
        elif self.reconstruction_loss == "mae":
            reconstruction_loss = mae
        elif self.reconstruction_loss == "mse":
            reconstruction_loss = mse
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        # classification loss
        class_loss = F.cross_entropy(class_logits, y, reduction="none")

        standard_loss = (
            reconstruction_loss + self.classifier * self.cls_loss_coeff * class_loss
        )
        detached_standard_loss = standard_loss.detach()

        # calculate entoropy and log_prob
        entropy = torch.zeros(n, device=x.device)
        log_prob = torch.zeros(n, device=x.device)

        for i in range(self.max_len):
            not_eosed = (i < message_length).float()
            entropy += speaker_outputs.entropy[:, i] * not_eosed
            log_prob += speaker_outputs.logits[:, i] * not_eosed
        entropy /= message_length.float()

        weighted_entropy = entropy.mean() * self.entropy_coeff

        # policy loss
        if self.wo_policy_loss:
            policy_loss = 0
        else:
            policy_loss = (
                (detached_standard_loss - self.baseline.predict(detached_standard_loss))
                * log_prob
            ).mean()

        # length policy loss
        length_loss = message_length.float() * self.length_const
        length_policy_loss = (
            (length_loss - self.len_baseline.predict(length_loss)) * log_prob
        ).mean()

        # total loss
        loss = (
            standard_loss.mean() + policy_loss + length_policy_loss - weighted_entropy
        )

        self.log(f"{log_header}_length", message_length.float().mean(), sync_dist=True)
        self.log(f"{log_header}_bce", bce.mean(), sync_dist=True)
        self.log(f"{log_header}_mae", mae.mean(), sync_dist=True)
        self.log(f"{log_header}_mse", mse.mean(), sync_dist=True)

        self.log(
            f"{log_header}_reconstruction_loss",
            reconstruction_loss.mean(),
            sync_dist=True,
        )
        if self.classifier:
            self.log(f"{log_header}_class_loss", class_loss.mean(), sync_dist=True)
        self.log(f"{log_header}_standard_loss", standard_loss.mean(), sync_dist=True)

        self.log(f"{log_header}_entropy", entropy.mean(), sync_dist=True)
        if not self.wo_policy_loss:
            self.log(f"{log_header}_policy_loss", policy_loss)
        if self.length_const > 0:
            self.log(
                f"{log_header}_length_policy_loss", length_policy_loss, sync_dist=True
            )
            self.log(
                f"{log_header}_length_baseline",
                self.len_baseline.predict(length_loss),
                sync_dist=True,
            )

        self.log(f"{log_header}_loss", loss, sync_dist=True)
        self.log(
            f"{log_header}_baseline",
            self.baseline.predict(detached_standard_loss),
            sync_dist=True,
        )

        if self.training:
            self.baseline.update(detached_standard_loss)
            self.len_baseline.update(length_loss)

        if not save_samples:
            return loss

        channel, height, width = self.image_size

        table = wandb.Table(columns=["message", "label", "input", "output"])
        message = speaker_outputs.sequence.cpu().numpy()
        labels = y.cpu().numpy()
        inputs = x.reshape(n, channel, height, width).cpu().numpy()
        outputs = x_hat.reshape(n, channel, height, width).cpu().detach().numpy()

        _id = 0
        for m, l, i, o in zip(message, labels, inputs, outputs):
            if channel == 1:
                i = wandb.Image(i)
                o = wandb.Image(o)
            else:
                i = wandb.Image(i.transpose(1, 2, 0))
                o = wandb.Image(o.transpose(1, 2, 0))
            table.add_data(m, self.labels[l], i, o)
            _id += 1
            if _id >= self.images_per_batch:
                break
        self.logger.experiment.log({f"{log_header}_samples": table})

        return loss


def create_parser():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument("--gpu", nargs="*", default=None, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--limit_batches", type=float, default=None)

    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--speaker_lr", type=float, default=5e-5)
    parser.add_argument("--listener_lr", type=float, default=1e-4)
    parser.add_argument("--generator_lr", type=float, default=1e-5)
    parser.add_argument(
        "--reconstruction_loss", type=str, default="mae", choices=["bce", "mse", "mae"]
    )
    parser.add_argument("--wo_pretrain", action="store_true")
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)

    # Game parameters
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=2)
    parser.add_argument("--length_cost", type=float, default=0.0)

    # Model parameters
    parser = backbone_parser(parser)
    parser.add_argument(
        "--arch", type=str, default="transformer", choices=["rnn", "transformer"]
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="hidden dimension for RNN and d_model for Transformer",
    )
    # RNN parameters
    parser.add_argument("--embed_dim", type=int, default=10)
    # Transformer parameters
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
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

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="generate")
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
