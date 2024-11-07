import argparse
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

from src.data import (
    CocoDetectionDataModule,
    mnistClassLabels,
    cifar10ClassLabels,
)
from src.models.backbone.load import get_backbone
from src.models import (
    backbone_parser,
    Backbone,
    TransformerSpeakerListener,
    SpeakerOutputs,
    InstanceIdentifier,
)
from src.utils import (
    load_artifact,
    SaveCrossAttentionMapMixin,
    heatmap,
    calc_simclr_loss,
)


def main(config, sweep=False):
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    offline = config.offline

    # load dataset
    num_workers = os.cpu_count() if config.num_workers is None else config.num_workers
    num_workers = 32 if num_workers > 32 else num_workers
    if config.dataset == "coco_detections":
        dm = CocoDetectionDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            smallest_smaller=config.smallest_smaller,
            max_smallers_num=config.max_smallers_num,
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
        monitor="val_loss",
        mode="min",
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
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

    module = InstanceModule(
        backbone_lr=config.backbone_lr,
        speaker_lr=config.speaker_lr,
        listener_lr=config.listener_lr,
        identifier_lr=config.identifier_lr,
        max_len=config.max_len,
        smaller_max_len=config.smaller_max_len,
        speaker_dim=config.speaker_dim,
        simclr_dim=config.simclr_dim,
        identifier_dim=config.identifier_dim,
        vocab_size=config.vocab_size,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        arch=config.arch,
        nhead=config.nhead,
        dropout=config.dropout,
        num_layers=config.num_layers,
        straight_through=config.straight_through,
        detach_message=config.detach_message,
        similarity=config.similarity,
        cosine_temperature=config.cosine_temperature,
        backbone=config.backbone,
        dataset=config.dataset,
        backbone_checkpoint=backbone_checkpoint_path,
        freeze_backbone=not config.wo_freeze,
        show_last_attention=config.show_last_attention,
        instance_identification=config.instance_identification,
        simclr=config.simclr,
        only_positive_samples=config.only_positive_samples,
        sweep=sweep,
        instance_rate=config.instance_rate,
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
        module = InstanceModule.load_from_checkpoint(val_checkpoint.best_model_path)
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


class InstanceModule(L.LightningModule, SaveCrossAttentionMapMixin):
    def __init__(
        self,
        backbone_lr: float,
        speaker_lr: float,
        listener_lr: float,
        identifier_lr: float,
        max_len: int,
        smaller_max_len: int,
        speaker_dim: int,
        simclr_dim: int,
        identifier_dim: int,
        vocab_size: int,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        arch: str,
        nhead: int,
        dropout: float,
        num_layers: int,
        straight_through: bool,
        detach_message: bool,
        similarity: str,
        cosine_temperature: float,
        backbone: str,
        dataset: str,
        freeze_backbone: bool,
        sweep: bool,
        backbone_checkpoint: Optional[Path] = None,
        backbone_dim: Optional[int] = None,
        show_last_attention: bool = False,
        instance_identification: int = 1,
        only_positive_samples: bool = False,
        simclr: bool = False,
        instance_rate: float = 1.0,
    ):
        super(InstanceModule, self).__init__()
        self.save_hyperparameters()

        self.backbone_lr = backbone_lr
        self.speaker_lr = speaker_lr
        self.listener_lr = listener_lr
        self.identidier_lr = identifier_lr
        self.sweep = sweep
        self.max_len = max_len
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.similarity = similarity
        self.cosine_temperature = cosine_temperature
        self.show_last_attention = show_last_attention
        self.dataset = dataset
        self.instance_identification = instance_identification
        self.simclr = simclr
        self.instance_rate = instance_rate

        backbone: Backbone = get_backbone(
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
        elif dataset == "cifar10":
            self.image_size = (3, 32, 32)
            class_num = 10
            self.labels = cifar10ClassLabels()
        elif dataset == "coco_captions" or dataset == "coco_detections":
            self.image_size = (3, 224, 224)
            class_num = 0
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if arch == "rnn":
            raise Exception("Not implemented for RNN")
        elif arch == "transformer":
            self.simclr_model = TransformerSpeakerListener(
                backbone=backbone,
                max_len=max_len,
                speaker_dim=speaker_dim,
                listener_dim=simclr_dim,
                vocab_size=vocab_size,
                nhead=nhead,
                dropout=dropout,
                num_layers=num_layers,
                gumbe_softmax=True,
                straight_through=straight_through,
                detach_message=detach_message,
                listener_agent=nn.Identity(),
                freeze=freeze_backbone,
            )
            self.instance_identifier_model = InstanceIdentifier(
                backbone=self.simclr_model.backbone,
                speaker=self.simclr_model.speaker,
                d_model=identifier_dim,
                larger_max_len=max_len,
                smaller_max_len=smaller_max_len,
                detach_message=detach_message,
                vocab_size=vocab_size,
                nhead=nhead,
                dropout=dropout,
                num_layers=num_layers,
                only_positive_samples=only_positive_samples,
            )
            if show_last_attention:
                self.set_hook(self.simclr_model.speaker.transformer)

        else:
            raise ValueError(f"Unknown architecture: {arch}")
        self.baseline = MeanBaseline()
        self.len_baseline = MeanBaseline()
        self.lazy_baseline = MeanBaseline()

    def training_step(self, batch, batch_idx):
        return self._step(
            batch,
            batch_idx,
            log_header="train",
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
                {
                    "params": self.simclr_model.backbone.parameters(),
                    "lr": self.backbone_lr,
                },
                {
                    "params": self.simclr_model.speaker.parameters(),
                    "lr": self.speaker_lr,
                },
                {
                    "params": self.simclr_model.listener.parameters(),
                    "lr": self.listener_lr,
                },
                {
                    "params": self.instance_identifier_model.listener.parameters(),
                    "lr": self.identidier_lr,
                },
            ]
        )

    def forward(self, x):
        return self.simclr_model(x)

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_samples=False,
    ):
        simclr, larger, smallers, y = batch
        instance_n = int(larger.size(0) * self.instance_rate)
        larger = larger[:instance_n]
        smallers = (smallers[0][:instance_n], smallers[1][:instance_n])

        x_i, x_j, x_original = simclr
        x_original = [x_o.detach().cpu().numpy() for x_o in x_original]
        smallers_len, all_smallers = smallers
        n = x_i.size(0)

        # simclr loss
        if self.simclr:
            z_i, speaker_outputs_i = self.simclr_model(x_i)
            z_j, speaker_outputs_j = self.simclr_model(x_j)

            x = torch.cat([x_i, x_j], dim=0)
            z = torch.cat([z_i, z_j], dim=0)
            speaker_outputs = SpeakerOutputs.concat(
                [speaker_outputs_i, speaker_outputs_j]
            )
            message_length = find_lengths(speaker_outputs.sequence)

            simclr_loss, acc = calc_simclr_loss(
                z, n, self.similarity, self.cosine_temperature
            )
        else:
            simclr_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # instance identification loss
        if self.instance_identification != 0.0:
            instance_loss, larger_speaker_outputs, smaller_speaker_outputs = (
                self.instance_identifier_model(larger, all_smallers, smallers_len)
            )
        else:
            instance_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = simclr_loss.mean() + instance_loss * self.instance_identification

        if self.simclr:
            self.log(
                f"{log_header}_length", message_length.float().mean(), sync_dist=True
            )
            self.log(f"{log_header}_acc", acc.mean(), sync_dist=True)
            self.log(f"{log_header}_simclr_loss", simclr_loss.mean(), sync_dist=True)

        if self.instance_identification != 0.0:
            self.log(
                f"{log_header}_instance_loss", instance_loss.mean(), sync_dist=True
            )

        self.log(f"{log_header}_loss", loss, sync_dist=True)

        if not save_samples:
            return loss

        channel, height, width = self.image_size

        labels = y[0]
        originals = x_original
        xs_i = x_i.cpu().numpy()
        xs_j = x_j.cpu().numpy()
        msg_i = speaker_outputs.sequence[:n].cpu().numpy()
        msg_j = speaker_outputs.sequence[n:].cpu().numpy()

        table = wandb.Table(
            columns=[
                "label",
                "original",
                "image_i",
                "image_j",
                "msg_i",
                "msg_j",
            ]
        )
        _id = 0
        for lbl, org, x_i, x_j, m_i, m_j in zip(
            labels, originals, xs_i, xs_j, msg_i, msg_j
        ):
            if channel == 1:
                org = wandb.Image(org)
                x_i = wandb.Image(x_i)
                x_j = wandb.Image(x_j)
            else:
                org = wandb.Image(org.transpose(1, 2, 0))
                x_i = wandb.Image(x_i.transpose(1, 2, 0))
                x_j = wandb.Image(x_j.transpose(1, 2, 0))
            table.add_data(lbl, org, x_i, x_j, m_i, m_j)
            _id += 1
            if _id >= self.images_per_batch:
                break
        self.logger.experiment.log({"samples": table})

        return loss

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
    parser.add_argument("--identifier_lr", type=float, default=1e-4)
    parser.add_argument("--wo_freeze", action="store_true")

    # Game parameters
    parser.add_argument(
        "--dataset", type=str, default="coco_detections", choices=["coco_detections"]
    )
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=8)
    parser.add_argument("--smaller_max_len", type=int, default=4)
    # SimCLR parameters
    parser.add_argument(
        "--similarity", type=str, default="cosine", choices=["cosine", "dot"]
    )
    parser.add_argument("--cosine_temperature", type=float, default=0.1)
    parser.add_argument("--simclr", action="store_true")
    # instance identification parameters
    parser.add_argument("--instance_identification", type=float, default=1)
    parser.add_argument("--max_smallers_num", type=int, default=None)
    parser.add_argument("--smallest_smaller", type=int, default=None)
    parser.add_argument("--only_positive_samples", action="store_true")

    parser.add_argument("--instance_rate", type=float, default=1.0)

    # Model parameters
    parser = backbone_parser(parser)
    parser.add_argument(
        "--arch", type=str, default="transformer", choices=["rnn", "transformer"]
    )
    parser.add_argument("--speaker_dim", type=int, default=32)
    parser.add_argument("--simclr_dim", type=int, default=128)
    parser.add_argument("--identifier_dim", type=int, default=32)
    # Transformer parameters
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Optimization parameters
    parser.add_argument("--straight_through", action="store_true")
    parser.add_argument("--detach_message", action="store_true")

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="instance")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--images_per_batch", type=int, default=20)
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--show_last_attention", action="store_true")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
