from typing import Optional
from pathlib import Path

import torch.nn as nn
import wandb

from .parser import construct_backbone
from src.wandb.pretrain.train import PretrainBackboneModule

# WARNING: This file depends on src.wandb.pretrain.train.PretrainBackboneModule which also depends on src.models.
# WARNING: So, you should not incloud this file to src.models.
# WARNING: Otherwise, it will cause circular import error.


def get_backbone(
    dataset: str,
    name: str | None = None,
    pretrain: Path | None = None,
    freeze: bool = False,
    dim: Optional[int] = None,
) -> nn.Module:
    """this is a wrapper function to get the backbone model.
    When the pretrain is given, it loads the pretrained model.
    Otherwise, it constructs the backbone model.

    Args:
        dataset (str): _description_
        name (str | None, optional): _description_. Defaults to None.
        pretrain (str | None, optional): Normally, this is the checkpoint of the pretrained backbone model. But for Vision, this is the path to the pretrained model.  Defaults to None.
        batch_size (int, optional): Used only for pretraining Vision. Defaults to 32.
        freeze (bool, optional): Defaults to False.
        dim (Optional[int], optional): Sometimes being ignored. Defaults to None.
    Raises:
        ValueError: raise when unknown backbone name is given

    Returns:
        list[nn.Module, int]: backbone model and its output size
    """
    if pretrain is not None:
        return load_backbone(pretrain)
    return construct_backbone(dataset=dataset, name=name, freeze=freeze, dim=dim)


def load_backbone(checkpoint: Path) -> nn.Module:
    model = PretrainBackboneModule.load_from_checkpoint(
        checkpoint, map_location=lambda storage, loc: storage
    ).pretrain_net
    return model
