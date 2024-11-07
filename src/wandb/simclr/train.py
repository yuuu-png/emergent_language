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
    MNISTDataModule,
    CIFAR10DataModule,
    CocoCaptionsDataModule,
    mnistClassLabels,
    cifar10ClassLabels,
    SuperCLEVRDataModule,
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
            augmentation_min_scale=config.augmentation_min_scale,
        )
    elif config.dataset == "coco":
        dm = CocoCaptionsDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            simclr=True,
            augmentation=config.data_augmentation,
            min_crop=config.augmentation_min_scale,
            sub_min_crop=config.augmentation_sub_min_scale,
            sub_max_crop=config.augmentation_sub_max_scale,
        )
    elif config.dataset == "superCLEVR":
        dm = SuperCLEVRDataModule(
            train_batch_size=config.batch_size,
            val_batch_size=config.batch_size,
            num_workers=num_workers,
            simclr=True,
            augmentation=config.data_augmentation,
            min_crop=config.augmentation_min_scale,
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
        monitor="val_simclr_loss",
        mode="min",
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
    )
    save_n_epochs = ModelCheckpoint(
        filename="epoch-{epoch}-{step}",
        monitor="epoch",
        mode="max",
        every_n_epochs=config.save_n_epochs,
    )
    early_stopping = EarlyStopping(
        monitor="val_simclr_loss",
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
        + ([latest_checkpoint] if not sweep else [])
        + ([save_n_epochs] if config.save_n_epochs is not None else []),
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

    num_gpus = 1 if config.gpu is None else len(config.gpu)

    backbone_checkpoint_path = (
        load_artifact(
            artifact_id=config.backbone_checkpoint,
            wandb_logger=wandb_logger,
            global_rank=trainer.global_rank,
        )
        if config.backbone_checkpoint is not None
        else None
    )

    module = SimCLRModule(
        backbone_lr=config.backbone_lr,
        speaker_lr=config.speaker_lr,
        listener_lr=config.listener_lr,
        max_len=config.max_len,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        entropy_coeff=config.entropy_coeff,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        length_const=config.length_cost,
        arch=config.arch,
        nhead=config.nhead,
        dropout=config.dropout,
        num_layers=config.num_layers,
        gumbel_softmax=config.gumbel_softmax,
        straight_through=config.straight_through,
        wo_policy_loss=config.wo_policy_loss,
        detach_message=config.detach_message,
        similarity=config.similarity,
        cosine_temperature=config.cosine_temperature,
        backbone=config.backbone,
        dataset=config.dataset,
        backbone_checkpoint=backbone_checkpoint_path,
        freeze_backbone=not config.wo_freeze,
        lazy_speaker=config.lazy_speaker,
        lazy_speaker_beta1=config.lazy_speaker_beta1,
        lazy_speaker_beta2=config.lazy_speaker_beta2,
        classifier=config.classifier,
        classifier_lr=config.classifier_lr,
        classifier_hidden_dim=config.classifier_hidden_dim,
        cls_loss_coeff=config.cls_loss_coeff,
        show_last_attention=config.show_last_attention,
        num_gpus=num_gpus,
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
        module = SimCLRModule.load_from_checkpoint(val_checkpoint.best_model_path)
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


class SimCLRModule(L.LightningModule, SaveCrossAttentionMapMixin):
    class ListenrAgent(nn.Module):
        def __init__(
            self, input_dim: int, hidden_dim: int, class_num: int, dropout=0.1
        ):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, class_num),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = self.classifier(x)
            return x, logits

    def __init__(
        self,
        backbone_lr: float,
        speaker_lr: float,
        listener_lr: float,
        max_len: int,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        entropy_coeff: float,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        length_const: float,
        arch: str,
        nhead: int,
        dropout: float,
        num_layers: int,
        gumbel_softmax: bool,
        straight_through: bool,
        wo_policy_loss: bool,
        detach_message: bool,
        similarity: str,
        cosine_temperature: float,
        backbone: str,
        dataset: str,
        freeze_backbone: bool,
        lazy_speaker: bool,
        lazy_speaker_beta1: float,
        lazy_speaker_beta2: float,
        sweep: bool,
        backbone_checkpoint: Optional[Path] = None,
        backbone_dim: Optional[int] = None,
        classifier: bool = False,
        classifier_lr: float = 1e-4,
        classifier_hidden_dim: int = 64,
        cls_loss_coeff: int = 1,
        show_last_attention: bool = False,
        num_gpus: int = 4,
    ):
        super(SimCLRModule, self).__init__()
        self.save_hyperparameters()

        self.backbone_lr = backbone_lr
        self.speaker_lr = speaker_lr
        self.listener_lr = listener_lr
        self.sweep = sweep
        self.max_len = max_len
        self.entropy_coeff = entropy_coeff
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.length_const = length_const
        self.wo_policy_loss = wo_policy_loss
        self.similarity = similarity
        self.cosine_temperature = cosine_temperature
        self.lazy_speaker = lazy_speaker
        self.lazy_speaker_beta1 = lazy_speaker_beta1
        self.lazy_speaker_beta2 = lazy_speaker_beta2
        self.classifier = classifier
        self.classifier_lr = classifier_lr
        self.cls_loss_coeff = cls_loss_coeff
        self.show_last_attention = show_last_attention
        self.dataset = dataset
        self.num_gpus = num_gpus

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
        elif dataset == "coco":
            self.image_size = (3, 224, 224)
            class_num = 0
        elif dataset == "superCLEVR":
            self.image_size = (3, 224, 224)
            class_num = 0
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if arch == "rnn":
            self.model = RnnSpeakerListener(
                backbone=backbone,
                max_len=max_len,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                listener_agent=(
                    self.ListenrAgent(
                        input_dim=hidden_dim,
                        hidden_dim=classifier_hidden_dim,
                        class_num=class_num,
                    )
                    if classifier
                    else nn.Identity()
                ),
            )
        elif arch == "transformer":
            self.model = TransformerSpeakerListener(
                backbone=backbone,
                max_len=max_len,
                speaker_dim=hidden_dim,
                listener_dim=hidden_dim,
                vocab_size=vocab_size,
                nhead=nhead,
                dropout=dropout,
                num_layers=num_layers,
                gumbe_softmax=gumbel_softmax,
                straight_through=straight_through,
                detach_message=detach_message,
                listener_agent=(
                    self.ListenrAgent(
                        input_dim=hidden_dim,
                        hidden_dim=classifier_hidden_dim,
                        class_num=class_num,
                    )
                    if classifier
                    else nn.Identity()
                ),
                freeze=freeze_backbone,
            )
            if show_last_attention:
                self.set_hook(self.model.speaker.transformer)

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
                    "params": self.model.backbone.parameters(),
                    "lr": self.backbone_lr / self.num_gpus,
                },
                {
                    "params": self.model.speaker.parameters(),
                    "lr": self.speaker_lr / self.num_gpus,
                },
                {
                    "params": self.model.listener.parameters(),
                    "lr": self.listener_lr / self.num_gpus,
                },
                {
                    "params": self.model.listener_agent.parameters(),
                    "lr": self.classifier_lr / self.num_gpus,
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
        if self.dataset != "superCLEVR":
            x, y = batch
        else:
            x = batch
            # y is a tensor which has the same length as x
            y = torch.zeros(len(x[0]), device=x[0].device)
        x_i, x_j, x_original = x
        n = x_i.size(0)

        self.reset_attention_map()
        listener_outputs_i, speaker_outputs_i = self.model(x_i)
        attention_map_i = self.attention_map
        self.reset_attention_map()
        listener_outputs_j, speaker_outputs_j = self.model(x_j)
        attention_map_j = self.attention_map

        if self.classifier:
            z_i, class_logit_i = listener_outputs_i
            z_j, class_logit_j = listener_outputs_j
        else:
            z_i = listener_outputs_i
            z_j = listener_outputs_j

        x = torch.cat([x_i, x_j], dim=0)
        z = torch.cat([z_i, z_j], dim=0)
        speaker_outputs = SpeakerOutputs.concat([speaker_outputs_i, speaker_outputs_j])
        message_length = find_lengths(speaker_outputs.sequence)

        # standard loss
        # simclr loss
        if self.similarity == "cosine":
            similarity_f = nn.CosineSimilarity(dim=2)
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

        simclr_loss = F.cross_entropy(logits, labels, reduction="none") / 2
        acc = (torch.argmax(logits, dim=1) == labels).float()

        # classification loss
        # class_logits = torch.cat([class_logit_i, class_logit_j], dim=0)
        # ys = torch.cat([y, y], dim=0)
        # class_loss = F.cross_entropy(class_logits, ys, reduction="none")

        standard_loss = (
            simclr_loss  # + self.classifier * self.cls_loss_coeff * class_loss
        )
        detached_standard_loss = standard_loss.detach()

        # calculate entoropy and log_prob
        entropy = torch.zeros(2 * n, device=x.device)
        log_prob = torch.zeros(2 * n, device=x.device)

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

        # lazy speaker loss
        if self.lazy_speaker:
            alpha = self._lazy_speaker_alpha(acc)
            lazy_policy_loss = alpha * message_length
            lazy_policy_loss = (
                lazy_policy_loss - self.lazy_baseline.predict(lazy_policy_loss)
            ).mean()
        else:
            lazy_policy_loss = torch.tensor(0.0, device=x.device)

        # length policy loss
        length_loss = message_length.float() * self.length_const
        length_policy_loss = (
            (length_loss - self.len_baseline.predict(length_loss)) * log_prob
        ).mean()

        # total loss
        loss = (
            standard_loss.mean()
            + policy_loss
            + length_policy_loss
            + lazy_policy_loss
            - weighted_entropy
        )

        self.log(f"{log_header}_length", message_length.float().mean(), sync_dist=True)
        self.log(f"{log_header}_acc", acc.mean(), sync_dist=True)

        self.log(f"{log_header}_simclr_loss", simclr_loss.mean(), sync_dist=True)
        # if self.classifier:
        #     self.log(f"{log_header}_class_loss", class_loss.mean(), sync_dist=True)
        self.log(f"{log_header}_standard_loss", standard_loss.mean(), sync_dist=True)

        self.log(f"{log_header}_entropy", entropy.mean(), sync_dist=True)
        if not self.wo_policy_loss:
            self.log(f"{log_header}_policy_loss", policy_loss, sync_dist=True)
        if self.length_const > 0:
            self.log(
                f"{log_header}_length_policy_loss", length_policy_loss, sync_dist=True
            )
            self.log(
                f"{log_header}_length_baseline",
                self.len_baseline.predict(length_loss),
                sync_dist=True,
            )
        if self.lazy_speaker:
            self.log(f"{log_header}_alpha", alpha.mean(), sync_dist=True)
            self.log(f"{log_header}_lazy_policy_loss", lazy_policy_loss, sync_dist=True)
            self.log(
                f"{log_header}_lazy_baseline",
                self.lazy_baseline.predict(lazy_policy_loss),
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
            self.lazy_baseline.update(lazy_policy_loss)

        if not save_samples:
            return loss

        channel, height, width = self.image_size

        if self.dataset == "coco":
            labels = y[0]
        else:
            labels = y.cpu().numpy()
        if type(x_original) is list:
            originals = [o.cpu().numpy() for o in x_original]
        elif type(x_original) is torch.Tensor:
            originals = x_original.view(n, channel, height, width).cpu().numpy()
        else:
            raise Exception(
                f"x_original should be list or torch.Tensor but {type(x_original)}"
            )
        xs_i = x_i.cpu().numpy()
        xs_j = x_j.cpu().numpy()
        msg_i = speaker_outputs.sequence[:n].cpu().numpy()
        msg_j = speaker_outputs.sequence[n:].cpu().numpy()

        if self.show_last_attention:
            _, _, source_len = attention_map_i[-1].size()
            patch_size = int(math.sqrt(source_len))
            atten_map_seq_i = torch.zeros(n, self.max_len, channel, height, width)
            atten_map_seq_j = torch.zeros(n, self.max_len, channel, height, width)
            # 並列化したかったが，あきらめた
            for l in range(self.max_len):
                atten_map_seq_i[:, l] = heatmap(
                    x_i,
                    attention_map_i[-1][:, l].view(-1, patch_size, patch_size),
                )
                atten_map_seq_j[:, l] = heatmap(
                    x_j,
                    attention_map_j[-1][:, l].view(-1, patch_size, patch_size),
                )
            atten_map_seq_i = atten_map_seq_i.cpu().numpy()
            atten_map_seq_j = atten_map_seq_j.cpu().numpy()

            columns = [
                "label",
                "original",
                "image_i",
                "image_j",
                "msg_i",
                "msg_j",
            ]
            columns += [f"i_{i}" for i in range(self.max_len)]
            columns += [f"j_{i}" for i in range(self.max_len)]
            table = wandb.Table(columns=columns)
            _id = 0
            for lbl, org, x_i, x_j, m_i, m_j, a_i, a_j in zip(
                labels,
                originals,
                xs_i,
                xs_j,
                msg_i,
                msg_j,
                atten_map_seq_i,
                atten_map_seq_j,
            ):
                if channel == 1:
                    org = wandb.Image(org)
                    x_i = wandb.Image(x_i)
                    x_j = wandb.Image(x_j)
                else:
                    org = wandb.Image(org.transpose(1, 2, 0))
                    x_i = wandb.Image(x_i.transpose(1, 2, 0))
                    x_j = wandb.Image(x_j.transpose(1, 2, 0))
                a_i_ = []
                for i in range(self.max_len):
                    a_i_.append(wandb.Image(a_i[i].transpose(1, 2, 0)))
                a_j_ = []
                for i in range(self.max_len):
                    a_j_.append(wandb.Image(a_j[i].transpose(1, 2, 0)))
                if self.dataset == "coco":
                    l = lbl
                else:
                    l = self.labels[lbl]
                table.add_data(l, org, x_i, x_j, m_i, m_j, *a_i_, *a_j_)
                _id += 1
                if _id >= self.images_per_batch:
                    break
            self.logger.experiment.log({"samples": table})
        else:
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
                if hasattr(self, "labels"):
                    l = self.labels[lbl]
                else:
                    l = lbl
                table.add_data(l, org, x_i, x_j, m_i, m_j)
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

    parser.add_argument("--backbone_lr", type=float, default=1e-4 * 4)
    parser.add_argument("--speaker_lr", type=float, default=2e-5 * 4)
    parser.add_argument("--listener_lr", type=float, default=1e-4 * 4)
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)
    parser.add_argument("--augmentation_sub_min_scale", type=float, default=None)
    parser.add_argument("--augmentation_sub_max_scale", type=float, default=None)

    # Game parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
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

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="simclr")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--images_per_batch", type=int, default=20)
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--show_last_attention", action="store_true")
    parser.add_argument("--save_n_epochs", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
