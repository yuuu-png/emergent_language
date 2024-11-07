import itertools
from typing import Optional

import numpy as np
import random
from pathlib import Path
import torch
import torch.nn.functional as F

from pytorch_lightning.loggers import WandbLogger
import wandb


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def pad_mask(
    x: torch.Tensor, batch_size: int, eos_index: int, max_token_len: int
) -> torch.Tensor:
    """padding mask for transformer

    if the token is after the eos token, the mask is True.

    Args:
        x (torch.Tensor): the message from the speaker (max_token_len, batch_size, vocab_size)
        batch_size (int): batch size
        eos_index (int): eos index
        max_token_len (int): max token length

    Returns:
        torch.Tensor: padding mask (batch_size, max_token_len)
    """
    token_ids = torch.argmax(x, dim=-1)
    eos_indices = torch.where(token_ids == eos_index)
    mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=x.device)
    for eos_index, group in itertools.groupby(
        enumerate(eos_indices[1]), key=lambda x: x[1]
    ):
        group = list(group)
        mask[eos_index.item(), eos_indices[0][group[0][0]].item() + 1 :] = True
    return mask


def entropy(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


class MovingAverage:
    def __init__(self, size):
        self.window_size = size
        self.values = []
        self.average = 0.0
        self.num = 0

    def update(self, value):
        if self.num >= self.window_size:
            earliest = self.values.pop(0)
            self.average += (value - earliest) / self.num
        else:
            self.num += 1
            self.average += (value - self.average) / self.num
        self.values.append(value)
        return self.average

    def get(self):
        return self.average


def caption(words, alphabet=False, eos_index=0):
    if alphabet:
        words = np.array(list(map(chr, words + 97)))
    zero_index = np.where(words == eos_index)[0]
    if zero_index.size > 0:
        words = words[: zero_index[0]]
    return " ".join(map(str, words))


# from https://github.com/limacv/RGB_HSV_HSL
def hsv2rgb(hsv_h, hsv_l, hsv_s) -> torch.Tensor:
    hsv = torch.cat([hsv_h, hsv_s, hsv_l], dim=1)
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def heatmap(
    img: torch.Tensor,
    intensity: torch.Tensor,
    heatmap_conc: float = 0.4,
    normalize: str = "none",
    eps=1e-10,
) -> torch.Tensor:
    """create rgb heatmap

    Args:
        img (torch.Tensor): N, C=3, H, W
        intensity (torch.Tensor): N, h, w
        eps (_type_, optional): Defaults to 1e-10.

    Returns:
        torch.Tensor: N, C=3, H, W
    """
    N, C, H, W = img.shape
    _, h, w = intensity.shape

    # normalize and resize intensity
    if normalize == "linear":
        max_intensity = intensity.max(dim=-2, keepdim=True)[0].max(
            dim=-1, keepdim=True
        )[0]
        intensity = intensity / (max_intensity + eps)
    elif normalize == "softmax":
        intensity = intensity.reshape(N, h * w).softmax(dim=-1).reshape(N, h, w)
    elif normalize == "none":
        pass
    else:
        raise ValueError(
            f"normalize should be one of [linear, softmax, none], but got {normalize}"
        )
    intensity = intensity.unsqueeze(-3)
    intensity = F.interpolate(intensity, size=(H, W), mode="bilinear")
    intensity = intensity * 0.6 + 0.4

    heatmap = hsv2rgb(
        1 - intensity,
        torch.ones_like(intensity),
        torch.ones_like(intensity),
    )
    heatmap = heatmap * heatmap_conc + img * (1 - heatmap_conc)
    return heatmap


def load_artifact(
    artifact_id: str,
    wandb_logger: WandbLogger,
    global_rank: int,
    file_name: str = "model.chpt",
) -> Path:
    """load artifact in multi-gpu setting

    Args:
        artifact_id (str): id of the artifact
        wandb_logger (WandbLogger): wandb logger of pytorch lightning
        global_rank (int): trainer.global_rank

    Returns:
        Path: path to the downloaded artifact file
    """
    if global_rank == 0:
        artifact: wandb.Artifact = wandb_logger.use_artifact(
            artifact_id, artifact_type="model"
        )
        artifact_dir = artifact.download()
        artifact_path = Path(artifact_dir) / file_name
    else:
        artifact_path = Path("artifacts") / f"{artifact_id.split('/')[-1]}/{file_name}"

    return artifact_path
