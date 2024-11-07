import argparse
import os
import sys

import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
import wandb

from src.data import MNISTDataModule, CIFAR10DataModule
from src.models.backbone.load import get_backbone
from src.models import (
    backbone_parser,
    SimCLR,
)


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
            simclr=True,
            augmentation=config.data_augmentation,
        )
    elif config.dataset == "cifar10":
        dm = CIFAR10DataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            simclr=True,
            augmentation=config.data_augmentation,
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
    # to suppress lazy loading
    _ = wandb_logger.experiment

    # training settings
    val_checkpoint = ModelCheckpoint(
        filename="best-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_standard_loss",
        mode="min",
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_standard_loss",
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
    )

    module = ContinuousSimCLRModule(
        backbone_lr=config.backbone_lr,
        speaker_lr=config.speaker_lr,
        listener_lr=config.listener_lr,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        similarity=config.similarity,
        cosine_temperature=config.cosine_temperature,
        backbone=config.backbone,
        dataset=config.dataset,
        backbone_checkpoint=config.backbone_checkpoint,
        freeze_backbone=not config.wo_freeze,
        h_dim=config.h_dim,
        z_dim=config.z_dim,
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
    )

    if not config.fast_dev_run:
        module = ContinuousSimCLRModule.load_from_checkpoint(
            val_checkpoint.best_model_path
        )
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


class ContinuousSimCLRModule(L.LightningModule):
    def __init__(
        self,
        backbone_lr: float,
        speaker_lr: float,
        listener_lr: float,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        similarity: str,
        cosine_temperature: float,
        backbone: str,
        dataset: str,
        backbone_checkpoint: str,
        freeze_backbone: bool,
        h_dim: int,
        z_dim: int,
        sweep: bool,
    ):
        super(ContinuousSimCLRModule, self).__init__()
        self.save_hyperparameters()

        self.backbone_lr = backbone_lr
        self.speaker_lr = speaker_lr
        self.listener_lr = listener_lr
        self.sweep = sweep
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.similarity = similarity
        self.cosine_temperature = cosine_temperature

        backbone = get_backbone(
            dataset=dataset,
            name=backbone,
            pretrain=backbone_checkpoint,
            freeze=freeze_backbone,
        )
        if dataset == "mnist":
            self.image_size = (1, 28, 28)
        elif dataset == "cifar10":
            self.image_size = (3, 32, 32)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.model = SimCLR(
            backbone=backbone,
            h_dim=h_dim,
            z_dim=z_dim,
            freeze_backbone=freeze_backbone,
        )

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
                {"params": self.model.backbone.parameters(), "lr": self.backbone_lr},
                {"params": self.model.speaker.parameters(), "lr": self.speaker_lr},
                {"params": self.model.listener.parameters(), "lr": self.listener_lr},
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
        x_i, x_j, x_original = x
        n = x_i.size(0)
        h_i, z_i = self.model(x_i)
        h_i, z_j = self.model(x_j)

        x = torch.cat([x_i, x_j], dim=0)
        z = torch.cat([z_i, z_j], dim=0)

        # standard loss
        if self.similarity == "cosine":
            similarity_f = torch.nn.CosineSimilarity(dim=2)
            similarity_matrix = (
                similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.cosine_temperature
            )
        elif self.similarity == "dot":
            similarity_matrix = z @ z.t()

        sim_i_j = torch.diag(similarity_matrix, n)
        sim_j_i = torch.diag(similarity_matrix, -n)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(2 * n, 1)

        mask = torch.ones(
            2 * n, 2 * n, dtype=torch.bool, device=x_i.device
        ).fill_diagonal_(0)

        negative_samples = similarity_matrix[mask].view(2 * n, -1)

        labels = torch.zeros(2 * n, dtype=torch.long, device=x_i.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        standard_loss = F.cross_entropy(logits, labels, reduction="none") / 2
        acc = (torch.argmax(logits, dim=1) == labels).float()
        detached_standard_loss = standard_loss.detach()

        # total loss
        loss = standard_loss.mean()

        self.log(f"{log_header}_acc", acc.mean(), sync_dist=True)
        self.log(f"{log_header}_standard_loss", standard_loss.mean(), sync_dist=True)
        self.log(f"{log_header}_loss", loss, sync_dist=True)

        if not save_samples:
            return loss

        channel, height, width = self.image_size

        table = wandb.Table(
            columns=[
                "label",
                "original",
                "logit",
                "image_i",
                "image_j",
            ]
        )
        labels = y.cpu().numpy()
        originals = x_original.view(n, channel, height, width).cpu().numpy()
        logits = logits.cpu().numpy()
        xs_i = x_i.cpu().numpy()
        xs_j = x_j.cpu().numpy()

        _id = 0
        for (
            lbl,
            org,
            lgt,
            x_i,
            x_j,
        ) in zip(labels, originals, logits, xs_i, xs_j):
            if channel == 1:
                org = wandb.Image(org)
                x_i = wandb.Image(x_i)
                x_j = wandb.Image(x_j)
            else:
                org = wandb.Image(org.transpose(1, 2, 0))
                x_i = wandb.Image(x_i.transpose(1, 2, 0))
                x_j = wandb.Image(x_j.transpose(1, 2, 0))
            table.add_data(lbl, org, lgt, x_i, x_j)
            _id += 1
            if _id >= self.images_per_batch:
                break
        self.logger.experiment.log({"samples": table})

        return loss


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
    parser.add_argument("--speaker_lr", type=float, default=5e-5)
    parser.add_argument("--listener_lr", type=float, default=1e-4)
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")

    # Game parameters
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    # SimCLR parameters
    parser.add_argument(
        "--similarity", type=str, default="cosine", choices=["cosine", "dot"]
    )
    parser.add_argument("--cosine_temperature", type=float, default=0.1)

    # Model parameters
    parser = backbone_parser(parser)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--z_dim", type=int, default=256)

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="simclr")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--images_per_batch", type=int, default=50)
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
