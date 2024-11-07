import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import wandb

from src.data import (
    EmergentLanguageAndCaptionDataModule,
)
from src.models import (
    backbone_parser,
    Seq2SeqTransformer,
)
from src.utils import load_artifact, SaveCrossAttentionMapMixin


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


TGT_PAD_IDX = 0


# elのpadがeosとかぶっていて悪いが，とりあえずこれで実装を進める
def create_mask(src: torch.Tensor, tgt: torch.Tensor, src_pad_idx=0, tgt_pad_idx=None):
    if tgt_pad_idx is None:
        tgt_pad_idx = TGT_PAD_IDX
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, src.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
        torch.bool
    )

    src_padding_mask = (src == src_pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == tgt_pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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

    # load dataset
    num_workers = os.cpu_count() if config.num_workers is None else config.num_workers
    num_workers = 32 if num_workers > 32 else num_workers
    dm = EmergentLanguageAndCaptionDataModule(
        train_batch_size=config.batch_size,
        val_batch_size=config.batch_size,
        num_workers=num_workers,
        wandb_logger=wandb_logger,
    )
    TGT_PAD_IDX = dm.nl_tokenizer.pad_token_id

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
    save_n_epochs = ModelCheckpoint(
        filename="epoch-{epoch}-{step}",
        monitor="epoch",
        mode="max",
        every_n_epochs=config.save_n_epochs,
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

    module = TranslateModule(
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        images_per_batch=config.images_per_batch,
        batches_to_log=config.batches_to_log,
        save_interval=config.save_interval,
        length_const=config.length_cost,
        nhead=config.nhead,
        dropout=config.dropout,
        num_layers=config.num_layers,
        similarity=config.similarity,
        dataset=config.dataset,
        lazy_speaker=config.lazy_speaker,
        lazy_speaker_beta1=config.lazy_speaker_beta1,
        lazy_speaker_beta2=config.lazy_speaker_beta2,
        classifier=config.classifier,
        cls_loss_coeff=config.cls_loss_coeff,
        show_last_attention=config.show_last_attention,
        num_gpus=num_gpus,
        sweep=sweep,
        nl_vocab_size=dm.nl_vocab_size,
        wo_embedding=config.wo_embedding,
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
        module = TranslateModule.load_from_checkpoint(val_checkpoint.best_model_path)
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


class TranslateModule(L.LightningModule, SaveCrossAttentionMapMixin):
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
        lr: float,
        hidden_dim: int,
        vocab_size: int,
        images_per_batch: int,
        batches_to_log: int,
        save_interval: int,
        length_const: float,
        nhead: int,
        dropout: float,
        num_layers: int,
        similarity: str,
        dataset: str,
        lazy_speaker: bool,
        lazy_speaker_beta1: float,
        lazy_speaker_beta2: float,
        sweep: bool,
        nl_vocab_size: int,
        classifier: bool = False,
        cls_loss_coeff: int = 1,
        show_last_attention: bool = False,
        num_gpus: int = 4,
        wo_embedding: bool = False,
    ):
        super(TranslateModule, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.sweep = sweep
        self.images_per_batch = images_per_batch
        self.batches_to_log = batches_to_log
        self.save_interval = save_interval
        self.length_const = length_const
        self.similarity = similarity
        self.lazy_speaker = lazy_speaker
        self.lazy_speaker_beta1 = lazy_speaker_beta1
        self.lazy_speaker_beta2 = lazy_speaker_beta2
        self.classifier = classifier
        self.cls_loss_coeff = cls_loss_coeff
        self.show_last_attention = show_last_attention
        self.dataset = dataset
        self.num_gpus = num_gpus

        self.model = Seq2SeqTransformer(
            num_decoder_layers=num_layers,
            num_encoder_layers=num_layers,
            emb_size=hidden_dim,
            nhead=nhead,
            src_vocab_size=vocab_size,
            tgt_vocab_size=nl_vocab_size + 1,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            wo_src_embedding=wo_embedding,
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
        src, tgt = batch
        tgt_input = tgt[:-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )
        logits = self.model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1), ignore_index=0
        )

        self.log(f"{log_header}_loss", loss, on_step=True, on_epoch=True)

        return loss


def create_parser():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument("--gpu", nargs="*", default=None, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--limit_batches", type=float, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wo_freeze", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--augmentation_min_scale", type=float, default=0.08)
    parser.add_argument("--augmentation_sub_min_scale", type=float, default=None)
    parser.add_argument("--augmentation_sub_max_scale", type=float, default=None)

    # Game parameters
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "coco"]
    )
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--length_cost", type=float, default=0.0)
    # SimCLR parameters
    parser.add_argument(
        "--similarity", type=str, default="cosine", choices=["cosine", "dot"]
    )

    # Model parameters
    parser = backbone_parser(parser)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="hidden dimension for RNN and d_model for Transformer",
    )
    # Transformer parameters
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Lazimpa
    parser.add_argument("--lazy_speaker", action="store_true")
    parser.add_argument("--lazy_speaker_beta1", type=float, default=45)
    parser.add_argument("--lazy_speaker_beta2", type=float, default=10)
    # Classifier parameters
    parser.add_argument("--classifier", action="store_true")
    parser.add_argument("--cls_loss_coeff", type=float, default=1)

    parser.add_argument("--wo_embedding", action="store_true")

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="translate")
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
