import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.backbone import BackboneWrapper, Backbone


class Sender(nn.Module):
    def __init__(self, visual_features_dim: int, output_dim: int):
        super(Sender, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(visual_features_dim, visual_features_dim),
            # this is in the BackboneWrapper
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
            nn.Linear(visual_features_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.fc(x)


class Receiver(nn.Module):
    def __init__(self, visual_features_dim: int, output_dim: int):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_features_dim, visual_features_dim),
            nn.BatchNorm1d(visual_features_dim),
            nn.ReLU(),
            nn.Linear(visual_features_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.fc(x)


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        h_dim: int,
        z_dim: int,
        freeze_backbone: bool,
    ):
        super(SimCLR, self).__init__()
        self.backbone = BackboneWrapper(
            backbone,
            output_size=backbone.dim,
            freeze=freeze_backbone,
        )
        self.speaker = Sender(visual_features_dim=backbone.dim, output_dim=h_dim)
        self.listener = Receiver(visual_features_dim=h_dim, output_dim=z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        # if x is [feature_vecotor for encoder, feature_map], then x[1] is the feature map
        # this is the case that the encoder is a cnn like monotone28, color32
        # (batch_size, channels, height, width) -> (batch_size, channels, 1, 1)
        if isinstance(x, tuple):
            x = x[1]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        # when the encoder is vision transformer, the output is (num_patches, batch_size, hidden_dim)
        # so we need to reshape it to (batch_size, hidden_dim)
        elif x.dim() == 3:
            x = x.mean(dim=0)
        h = self.speaker(x)
        z = self.listener(h)
        return h, z
