from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as vision_models

from .backbone import Monotone28, Color32, ViT_wrapper, DINO_wrapper, ResNetWrapper


def backbone_parser(parser: ArgumentParser, wo_backbone_checkpoint: bool = False):
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="""
This is the backbone of the vision model.
This is different from the dataset:
MNIST: monotone28.
Cifar10: pretrained_resnet50(default), resnetXX(XX=18, 34, 50, 101, 152), pretrained_resnetXX, color32.
ImageNet: vit_X_XX, pretrained_vit_X_XX (vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14), resnetXX, pretrained_resnetXX.
""",
    )
    if not wo_backbone_checkpoint:
        parser.add_argument(
            "--backbone_checkpoint",
            type=str,
            default=None,
            help="""
    This is the checkpoint of the pretrained backbone model.""",
        )
    return parser


def construct_backbone(
    dataset: str,
    name: str | None = None,
    freeze: bool = False,
    dim: Optional[int] = None,
    use_attention: bool = False,
) -> nn.Module:
    """_summary_

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
    if name == None:
        if dataset == "mnist":
            name = "monotone28"
        elif dataset == "cifar10":
            name = "color32"
        else:
            raise ValueError(f"Unknown dataset : {dataset}")

    if name == "monotone28":
        if dim is None:
            dim = 32
        model = Monotone28(dim=dim)
    elif name == "color32":
        if dim is None:
            dim = 32
        model = Color32(dim=dim)
    elif name == "resnet18":
        model = ResNetWrapper(vision_models.resnet18())
    elif name == "resnet34":
        model = ResNetWrapper(vision_models.resnet34())
    elif name == "resnet50":
        model = ResNetWrapper(vision_models.resnet50())
    elif name == "resnet101":
        model = ResNetWrapper(vision_models.resnet101())
    elif name == "resnet152":
        model = ResNetWrapper(vision_models.resnet152())
    elif name == "pretrained_resnet18":
        model = ResNetWrapper(
            vision_models.resnet18(weights=vision_models.ResNet18_Weights.DEFAULT)
        )
    elif name == "pretrained_resnet34":
        model = ResNetWrapper(
            vision_models.resnet34(weights=vision_models.ResNet34_Weights.DEFAULT)
        )
    elif name == "pretrained_resnet50":
        model = ResNetWrapper(
            vision_models.resnet50(weights=vision_models.ResNet50_Weights.DEFAULT)
        )
    elif name == "pretrained_resnet101":
        model = ResNetWrapper(
            vision_models.resnet101(weights=vision_models.ResNet101_Weights.DEFAULT)
        )
    elif name == "pretrained_resnet152":
        model = ResNetWrapper(
            vision_models.resnet152(weights=vision_models.ResNet152_Weights.DEFAULT)
        )
    elif name == "vit_b_16":
        model = ViT_wrapper(vision_models.vit_b_16())
    elif name == "vit_b_32":
        model = ViT_wrapper(vision_models.vit_b_32())
    elif name == "vit_l_16":
        model = ViT_wrapper(vision_models.vit_l_16())
    elif name == "vit_l_32":
        model = ViT_wrapper(vision_models.vit_l_32())
    elif name == "vit_h_14":
        model = ViT_wrapper(vision_models.vit_h_14())
    elif name == "pretrained_vit_b_16":
        weight = vision_models.ViT_B_16_Weights.DEFAULT
        model = ViT_wrapper(
            vision_models.vit_b_16(weights=weight), preprocess=weight.transforms()
        )
    elif name == "pretrained_vit_b_32":
        weight = vision_models.ViT_B_32_Weights.DEFAULT
        model = ViT_wrapper(
            vision_models.vit_b_32(weights=weight), preprocess=weight.transforms()
        )
    elif name == "pretrained_vit_l_16":
        weight = vision_models.ViT_L_16_Weights.DEFAULT
        model = ViT_wrapper(
            vision_models.vit_l_16(weights=weight), preprocess=weight.transforms()
        )
    elif name == "pretrained_vit_l_32":
        weight = vision_models.ViT_L_32_Weights.DEFAULT
        model = ViT_wrapper(
            vision_models.vit_l_32(weights=weight), preprocess=weight.transforms()
        )
    elif name == "pretrained_vit_h_14":
        weight = vision_models.ViT_H_14_Weights.DEFAULT
        model = ViT_wrapper(
            vision_models.vit_h_14(weights=weight), preprocess=weight.transforms()
        )
    elif name == "dino_s_16":
        weight = vision_models.ViT_B_16_Weights.DEFAULT
        model = DINO_wrapper(
            "dinov2_vits14_reg", preprocess=weight.transforms(), attention=use_attention
        )
    elif name == "dino_b_16":
        weight = vision_models.ViT_B_16_Weights.DEFAULT
        model = DINO_wrapper(
            "dinov2_vitb14_reg", preprocess=weight.transforms(), attention=use_attention
        )
    elif name == "dino_l_16":
        weight = vision_models.ViT_B_16_Weights.DEFAULT
        model = DINO_wrapper(
            "dinov2_vitl14_reg", preprocess=weight.transforms(), attention=use_attention
        )
    elif name == "dino_g_16":
        weight = vision_models.ViT_B_16_Weights.DEFAULT
        model = DINO_wrapper(
            "dinov2_vitg14_reg", preprocess=weight.transforms(), attention=use_attention
        )
    else:
        raise ValueError(f"Unknown backbone name: {name}")

    # This is a hack to remove the final layer of the resnet model
    # The final linear layer is implemented in the model's forward method
    if "resnet" in name:
        model.fc = nn.Identity()

    if freeze is True:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model
