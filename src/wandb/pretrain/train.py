import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
import wandb

from src.data import MNISTDataModule, CIFAR10DataModule
from src.models import backbone_parser, construct_backbone
import src.models.generator as generators
from src.wandb.pretrain.utils import PretainTransformerBackboneNet


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
        )

    args = [arg.lstrip("-") for arg in sys.argv[1:]]
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
    early_stopping = EarlyStopping(
        monitor="val_reconstruction_loss",
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
        ],
        limit_predict_batches=config.limit_batches,
        limit_test_batches=config.limit_batches,
        limit_train_batches=config.limit_batches,
        limit_val_batches=config.limit_batches,
        log_every_n_steps=50 if num_batches > 50 else num_batches // 4,
    )

    module = PretrainBackboneModule(
        lr=config.lr,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        hidden_dim=config.hidden_dim,
        dataset=config.dataset,
        sweep=sweep,
        backbone=config.backbone,
    )

    trainer.fit(module, dm)

    # Because of DDP, there are processes of number of GPUs.
    if (config.gpu is None and torch.cuda.device_count() > 1) or (
        config.gpu is not None and len(config.gpu) > 1
    ):
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
        module = PretrainBackboneModule.load_from_checkpoint(
            val_checkpoint.best_model_path,
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
        wandb_logger.experiment.log_artifact(module_artifact)

    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


class PretrainBackboneModule(L.LightningModule):
    def __init__(
        self,
        lr: float,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        hidden_dim: int,
        dataset: str,
        sweep: bool,
        backbone: str,
    ):
        super(PretrainBackboneModule, self).__init__()
        self.lr = lr
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval

        self.save_hyperparameters()

        if dataset == "mnist":
            decoder = generators.Monotone28(z_dim=hidden_dim)
            self.image_size = (1, 28, 28)
        elif dataset == "cifar10":
            decoder = generators.Color32(z_dim=hidden_dim)
            self.image_size = (3, 32, 32)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        self.pretrain_net = construct_backbone(
            dataset=dataset, name=backbone, dim=hidden_dim
        )

        self.model = PretainTransformerBackboneNet(
            encoder=self.pretrain_net,
            encoder_output_channels=self.pretrain_net.dim,
            decoder_input_dim=hidden_dim,
            decoder=decoder,
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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        x_hat = self.model(x)

        mae = F.l1_loss(x_hat, x)
        reconstruction_loss = mae
        loss = reconstruction_loss

        self.log(f"{log_header}_mae", mae.mean(), sync_dist=True)

        self.log(
            f"{log_header}_reconstruction_loss",
            reconstruction_loss.mean(),
            sync_dist=True,
        )
        self.log(f"{log_header}_loss", loss, sync_dist=True)

        if not save_samples:
            return loss

        channel, height, width = self.image_size

        table = wandb.Table(columns=["label", "input", "output"])
        labels = y.cpu().numpy()
        inputs = x.view(n, channel, height, width).cpu().numpy()
        outputs = x_hat.view(n, channel, height, width).cpu().numpy()

        _id = 0
        for l, i, o in zip(labels, inputs, outputs):
            if channel == 1:
                i = wandb.Image(i)
                o = wandb.Image(o)
            else:
                i = wandb.Image(i.transpose(1, 2, 0))
                o = wandb.Image(o.transpose(1, 2, 0))
            table.add_data(l, i, o)
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--limit_batches", type=float, default=None)
    parser.add_argument("--resize", type=int, default=None)

    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="hidden dimension for RNN and d_model for Transformer",
    )
    parser = backbone_parser(parser, wo_backbone_checkpoint=True)

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="pretrain")
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
