from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vision_models


class Backbone(nn.Module, ABC):
    @property
    @abstractmethod
    def image_feature_size(self):
        """In transformer, the sequence length of the input image features"""
        pass

    @property
    @abstractmethod
    def dim(self):
        """In transformer, the dimension of the output of each feature vector of the backbone model"""
        pass


class BackboneWrapper(nn.Module):
    def __init__(self, backbone: Backbone, output_size, freeze):
        super(BackboneWrapper, self).__init__()
        self.fc = nn.Linear(backbone.dim, output_size)
        self.backbone = backbone
        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.fc(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x


class Monotone28(Backbone):
    def __init__(self, dim=32):
        super(Monotone28, self).__init__()
        self._dim = dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim // 2, dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def image_feature_size(self):
        return 49

    @property
    def dim(self):
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """extract future from image

        Args:
            x (torch.Tensor): 1*28*28 (MNSIT)

        Returns:
            torch.Tensor: 49*(batch_size)*(dim=32), (batch_size)*(dim=32)*7*7
        """
        x = self.cnn(x)
        feature_map = x
        x = x.flatten(start_dim=2)
        # x : (batch_size, dim, 49)
        x = x.permute(2, 0, 1)
        # x : (49, batch_size, dim)

        return x, feature_map


class Color32(Backbone):
    def __init__(self, dim=32):
        super(Color32, self).__init__()
        self._dim = dim

        self.cnn = nn.Sequential(
            nn.Conv2d(3, dim // 3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim // 3, (dim // 3) * 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d((dim // 3) * 2, dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def image_feature_size(self):
        return 64

    @property
    def dim(self):
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """extract future from image

        Args:
            x (torch.Tensor): 3*32*32 (CIFAR-10)

        Returns:
            torch.Tensor: 64*(batch_size)*(dim=32), (batch_size)*(dim=32)*8*8
        """
        x = self.cnn(x)
        feature_map = x
        x = x.flatten(start_dim=2)
        x = x.permute(2, 0, 1)
        return x, feature_map


class ResNetWrapper(Backbone):
    def __init__(
        self,
        resnet: vision_models.ResNet,
        preprocess: nn.Module = nn.Identity(),
    ) -> None:
        super(ResNetWrapper, self).__init__()
        self.resnet = resnet
        self.preprocess = preprocess

    @property
    def image_feature_size(self):
        return 1

    @property
    def dim(self):
        return self.resnet.fc.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the all the hidden states of the vision transformer

        This code is copy pasted from the vision transformer implementation in torchvision
        before the final layer is applied.

        Args:
            x (torch.Tensor): input image of shape (batch_size, 3, image_size, image_size)

        Returns:
            torch.Tensor: output of the vision transformer
        """
        x = self.preprocess(x)
        x = self.resnet(x)
        x = x.unsqueeze(0)

        return x


class ViT_wrapper(Backbone):
    def __init__(
        self,
        base_vit: vision_models.VisionTransformer,
        preprocess: nn.Module = nn.Identity(),
    ) -> None:
        super(ViT_wrapper, self).__init__()
        self.base_vit = base_vit
        self.preprocess = preprocess

    @property
    def image_feature_size(self):
        return self.base_vit.seq_length

    @property
    def dim(self):
        return self.base_vit.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the all the hidden states of the vision transformer

        This code is copy pasted from the vision transformer implementation in torchvision
        before the final layer is applied.

        Args:
            x (torch.Tensor): input image of shape (batch_size, 3, image_size, image_size)

        Returns:
            torch.Tensor: output of the vision transformer
        """
        x = self.preprocess(x)
        # Reshape and permute the input tensor
        x = self.base_vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.base_vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.base_vit.encoder(x)

        # x : (batch_size, num_patches + 1, hidden_dim)
        x = x.permute(1, 0, 2)
        # x : (num_patches + 1, batch_size, hidden_dim)

        return x


class DINO_wrapper(Backbone):
    def attention_hook(self, module, input, output) -> None:
        # (batch_size, num_heads, num_patches + 1, num_patches + 1)
        self.attention_map = input[0]

    def __init__(
        self,
        model: str,
        cls: bool = False,
        attention: bool = False,
        preprocess: nn.Module = nn.Identity(),
    ) -> None:
        super(DINO_wrapper, self).__init__()
        self.base_vit = torch.hub.load("facebookresearch/dinov2", model)
        self.cls = cls
        self.attention = attention
        if attention:
            # see Attention and MemEffAttention
            # on https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation_m2f/models/backbones/vit.py
            self.base_vit.blocks[-1].attn.attn_drop.register_forward_hook(
                self.attention_hook
            )
        self.preprocess = preprocess

    @property
    def image_feature_size(self):
        return self.base_vit.num_patches + 1 if self.cls else self.base_vit.num_patches

    @property
    def dim(self):
        return self.base_vit.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the all the hidden states of the vision transformer

        This code is copy pasted from the vision transformer implementation in torchvision
        before the final layer is applied.

        Args:
            x (torch.Tensor): input image of shape (batch_size, 3, image_size, image_size)

        Returns:
            torch.Tensor: output of the vision transformer
        """
        x = self.preprocess(x)
        results = self.base_vit.forward_features(x)
        if self.cls:
            x = torch.concat(
                [results["x_norm_clstoken"], results["x_norm_patchtokens"]], dim=0
            )
        else:
            x = results["x_norm_patchtokens"]

        if self.attention:
            x = torch.cat([x, self.attention_map], dim=1)

        x = x.permute(1, 0, 2)
        # print(self.attention_map)

        return x
