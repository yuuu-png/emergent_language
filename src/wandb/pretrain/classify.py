from argparse import ArgumentParser
from pathlib import Path
import os
import re
import sys

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.mnist import MNISTDataModule
from src.data.cifar10 import CIFAR10DataModule
from src.wandb.pretrain.train import PretrainBackboneModule


def main(config):
    L.seed_everything(config.seed)
    torch.set_float32_matmul_precision("medium")

    # load the data
    num_workers = os.cpu_count() if config.num_workers is None else config.num_workers
    num_workers = 32 if num_workers > 32 else num_workers
    if config.dataset == "mnist":
        dm = MNISTDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
        )
    elif config.dataset == "cifar10":
        dm = CIFAR10DataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

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
        offline=config.offline,
        name=name,
        project=config.project,
        save_dir="wandb-logs",
        log_model=True,
        tags=args,
    )
    wandb_logger.experiment.config.update(config)

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
    traienr = L.Trainer(
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
    artifact = wandb_logger.use_artifact(config.checkpoint, artifact_type="model")
    artifact_dir = artifact.download()
    checkpoint = Path(artifact_dir) / "model.chpt"
    simclrModule = PretrainBackboneModule.load_from_checkpoint(
        checkpoint, map_location=lambda storage, loc: storage
    )

    module = ClassifyModule(
        trained_module=simclrModule,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        images_per_batch=config.images_per_batch,
        lr=config.lr,
        num_classes=10,
    )
    traienr.fit(module, datamodule=dm)
    if not config.fast_dev_run:
        module = ClassifyModule.load_from_checkpoint(
            val_checkout.best_model_path, model=simclrModule
        )
    traienr.test(module, datamodule=dm)

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

    class Pretrain(torch.nn.Module):
        def __init__(self, net, classifier):
            super().__init__()
            self.net = net
            self.classifier = classifier

        def forward(self, x):
            with torch.no_grad():
                _, x = self.net(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def __init__(
        self,
        trained_module: PretrainBackboneModule,
        batches_to_log: int,
        save_interval: int,
        images_per_batch: int,
        lr: float,
        num_classes: int,
    ):
        super().__init__()
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.images_per_batch = images_per_batch
        self.lr = lr

        hidden_dim = trained_module.model.fc.in_features
        self.model = self.Pretrain(
            trained_module.model.encoder,
            self.MLP(hidden_dim, hidden_dim, num_classes),
        )
        for param in self.model.net.parameters():
            param.requires_grad = False
        self.model.net.eval()

        self.save_hyperparameters(ignore="module")
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
        self.logger.experiment.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    preds=self.conf_mat_pred,
                    y_true=self.conf_mat_y_true,
                    class_names=[
                        "airplane",
                        "automobile",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck",
                    ],
                )
            }
        )

    def test_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, "test", save_samples=False)

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), self.lr)

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_samples: False,
        conf_mat: bool = False,
    ):
        X, Y = batch
        logit = self.model(X)
        prob = F.softmax(logit, dim=-1)
        loss = F.cross_entropy(prob, Y)
        pred = torch.argmax(logit, dim=-1)
        acc = self.accuracy(pred, Y)

        self.log(f"{log_header}_loss", loss)
        self.log(f"{log_header}_acc", acc)

        if conf_mat:
            self.conf_mat_pred = np.append(self.conf_mat_pred, pred.cpu().numpy())
            self.conf_mat_y_true = np.append(self.conf_mat_y_true, Y.cpu().numpy())

        if not save_samples:
            return loss

        columns = ["original", "truth", "pred", "logit"]
        table = wandb.Table(columns=columns)
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()

        _id = 0
        for x, y, prb, prd in zip(X, Y, prob, pred):
            if x.shape[0] == 1:
                x = wandb.Image(x)
            else:
                x = wandb.Image(x.transpose(1, 2, 0))
            table.add_data(x, y, prd, prb)
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="if batch size is smaller than 1024, you should set this to 50",
    )
    parser.add_argument("--gpu", nargs="*", default=None, type=int)

    # game settings
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )

    # model settings
    parser.add_argument("--checkpoint", type=str, required=True)

    # wandb
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--project", type=str, default="simclr_classify")
    parser.add_argument("--offline", action="store_true")
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
