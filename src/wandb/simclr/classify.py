from argparse import ArgumentParser
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.mnist import MNISTDataModule, mnistClassLabels
from src.data.cifar10 import CIFAR10DataModule, cifar10ClassLabels
from src.wandb.simclr.train import SimCLRModule
from src.utils.utils import load_artifact


def main(config):
    L.seed_everything(config.seed)
    torch.set_float32_matmul_precision("medium")

    args = [
        arg.lstrip("-")
        for arg in sys.argv[1:]
        if re.search(r"checkpoint|gpu", arg) is None and len(arg) < 64
    ]
    if config.name is not None:
        name = config.name
    elif len(args) > 0:
        name = ",".join(args[:2])
    else:
        name = None
    wandb_logger = WandbLogger(
        name=name,
        project=config.project,
        save_dir="wandb-logs",
        log_model=True,
        tags=args,
        config=config,
    )

    # trainer settings
    val_checkout = ModelCheckpoint(
        filename="best-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_acc",
        mode="max",
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=config.early_stopping_patience
    )
    trainer = L.Trainer(
        devices="auto" if config.gpu is None else config.gpu,
        logger=wandb_logger,
        max_epochs=config.epoch,
        fast_dev_run=config.fast_dev_run,
        deterministic=True,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            val_checkout,
            latest_checkpoint,
            early_stopping,
            ModelSummary(max_depth=2),
        ],
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
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

    module = ClassifyModule(
        trained_module=simclrModule,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        images_per_batch=config.images_per_batch,
        train_whole_listener=config.train_whole_listener,
        listener_lr=config.listener_lr,
        classifier_lr=config.classifier_lr,
        class_labels=class_labels,
    )
    trainer.fit(module, datamodule=dm)
    if not config.fast_dev_run:
        module = ClassifyModule.load_from_checkpoint(
            val_checkout.best_model_path,
            model=simclrModule,
            trained_module=simclrModule,
        )
    trainer.test(module, datamodule=dm)

    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


class ClassifyModule(L.LightningModule):
    class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def __init__(
        self,
        trained_module: SimCLRModule,
        batches_to_log: int,
        save_interval: int,
        images_per_batch: int,
        train_whole_listener: bool,
        listener_lr: float,
        classifier_lr: float,
        class_labels: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="trained_module")

        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.images_per_batch = images_per_batch
        self.listener_lr = listener_lr
        self.classifier_lr = classifier_lr
        num_classes = len(class_labels)
        self.class_labels = class_labels

        self.model = trained_module.model
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.backbone.eval()
        for param in self.model.speaker.parameters():
            param.requires_grad = False
        self.model.speaker.eval()
        if not train_whole_listener:
            for param in self.model.listener.parameters():
                param.requires_grad = False
            self.model.listener.eval()
        output_dim = trained_module.hparams_initial["hidden_dim"]
        self.model.listener_agent = self.MLP(output_dim, output_dim, num_classes)

        self.accuracy = MulticlassAccuracy(num_classes=num_classes)

    def training_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, "train", save_samples=False)

    def on_validation_start(self):
        self.conf_mat_pred = np.array([])
        self.conf_mat_y_true = np.array([])

    def validation_step(self, batch, batch_idx: int):
        save_samples = (
            batch_idx < self.batches_to_log
            and self.current_epoch % self.save_interval == 0
        )
        return self._step(batch, batch_idx, "val", save_samples, conf_mat=True)

    def on_validation_end(self):
        if self.trainer.is_global_zero:
            self.logger.experiment.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        preds=self.conf_mat_pred,
                        y_true=self.conf_mat_y_true,
                        class_names=self.class_labels,
                    )
                }
            )

    def test_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, "test", save_samples=False)

    def configure_optimizers(self):
        return optim.Adam(
            [
                {"params": self.model.listener.parameters(), "lr": self.listener_lr},
                {
                    "params": self.model.listener_agent.parameters(),
                    "lr": self.classifier_lr,
                },
            ]
        )

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_samples: False,
        conf_mat: bool = False,
    ):
        X, Y = batch
        logit, speaker_output = self.model(X)
        prob = F.softmax(logit, dim=-1)
        loss = F.cross_entropy(prob, Y)
        pred = torch.argmax(logit, dim=-1)
        acc = self.accuracy(pred, Y)

        self.log(f"{log_header}_loss", loss, sync_dist=True)
        self.log(f"{log_header}_acc", acc, sync_dist=True)

        if conf_mat:
            self.conf_mat_pred = np.append(self.conf_mat_pred, pred.cpu().numpy())
            self.conf_mat_y_true = np.append(self.conf_mat_y_true, Y.cpu().numpy())

        if not save_samples:
            return loss

        columns = ["original", "truth", "pred", "logit", "tokens"]
        table = wandb.Table(columns=columns)
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()
        message = speaker_output.sequence.cpu().numpy()

        _id = 0
        for x, y, prb, prd, msg in zip(X, Y, prob, pred, message):
            if x.shape[0] == 1:
                x = wandb.Image(x)
            else:
                x = wandb.Image(x.transpose(1, 2, 0))
            table.add_data(x, y, prd, prb, msg)
            _id += 1
            if _id >= self.images_per_batch:
                break

        self.logger.experiment.log({"samples": table})


def create_parser():
    parser = ArgumentParser()
    # learning settings
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--train_whole_listener", action="store_true")
    parser.add_argument("--listener_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="if batch size is smaller than 1024, you should set this to 50",
    )
    parser.add_argument("--gpu", nargs="*", default=None, type=int)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)

    # game settings
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["mnist", "cifar10"]
    )

    # model settings
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--backbone_dim",
        type=int,
        default=None,
        help="this property is used when load a cnn backbone but need to specify its dim becouse this code can't tell it.",
    )

    # wandb
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--project", type=str, default="simclr_classify")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--images_per_batch", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    wandb.login()
    main(config=args)
