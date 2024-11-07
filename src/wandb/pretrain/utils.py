from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PretainTransformerBackboneNet(nn.Module):
    def __init__(
        self, encoder, decoder, encoder_output_channels, decoder_input_dim
    ) -> None:
        super(PretainTransformerBackboneNet, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder_output_channels, decoder_input_dim)
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tensor:
        # use feature map as input because it is the output of the cnn
        x = self.encoder(x)
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
        x = self.fc(x)
        x = F.relu(x)
        x = self.decoder(x)
        return x
