import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from egg.core import find_lengths
from .backbone.backbone import BackboneWrapper, Backbone
from .custom_transformer import CustomTransformerDecoderLayer


class SpeakerOutputs:
    def __init__(self, sequence, logits, entropy, one_hot_sequence):
        self.sequence = sequence
        self.logits = logits
        self.entropy = entropy
        self.one_hot_sequence = one_hot_sequence

    @staticmethod
    def concat(outputs, dim=0):
        sequence = torch.cat([output.sequence for output in outputs], dim=dim)
        logits = torch.cat([output.logits for output in outputs], dim=dim)
        entropy = torch.cat([output.entropy for output in outputs], dim=dim)
        one_hot_sequence = torch.cat(
            [output.one_hot_sequence for output in outputs], dim=dim
        )
        return SpeakerOutputs(sequence, logits, entropy, one_hot_sequence)


class TransformerSpeaker(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        vocab_size: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        gumbel_softmax: bool,
        straight_through: bool,
        image_feature_size: int,
    ):
        super(TransformerSpeaker, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.gumbel_softmax = gumbel_softmax
        self.straight_through = straight_through

        layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.image_pos_embedding = nn.Parameter(
            torch.randn(image_feature_size, 1, d_model)
        )
        self.token_pos_embedding = nn.Parameter(torch.randn(max_len, 1, d_model))

        self.bos_emb = nn.Parameter(torch.randn(1, d_model))
        self.embed = nn.Linear(vocab_size, d_model)
        self.token_classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self, image_features: torch.Tensor
    ) -> tuple[torch.FloatTensor, SpeakerOutputs]:
        """forward

        Args:
            image_features (torch.Tensor): len * batch_size * d_model

        Returns:
            tuple[torch.FloatTensor, SpeakerOutputs]: message, SpeakerOutputs
        """
        device = image_features.device
        batch_size = image_features.size(1)
        image_pos_position = self.image_pos_embedding.expand(-1, batch_size, -1)
        image_features += image_pos_position

        # because the sequence may end without eos token, we need to add 1 to max_len
        sequence = torch.zeros(
            (self.max_len + 1, batch_size), dtype=torch.long, device=device
        )
        logits = torch.zeros((self.max_len + 1, batch_size), device=device)
        entropy = torch.zeros((self.max_len + 1, batch_size), device=device)
        # make the all token to be eos token
        one_hot_sequence = torch.zeros(
            (self.max_len + 1, batch_size, self.vocab_size),
            device=device,
        )
        one_hot_sequence[:, :, 0] = 1

        # for bos token, we add 1 to max_len
        seq_hidden = torch.zeros(
            (self.max_len + 1, batch_size, self.d_model), device=device
        )
        prev_token_embedded = self.bos_emb.expand(batch_size, -1)

        token_pos_embedding = self.token_pos_embedding.expand(-1, batch_size, -1)
        for i in range(self.max_len):
            seq_hidden[i] = prev_token_embedded
            token_hidden = self.transformer(
                seq_hidden[: i + 1] + token_pos_embedding[: i + 1], image_features
            )
            token_hidden = token_hidden[-1]
            step_logits = F.log_softmax(self.token_classifier(token_hidden), dim=-1)
            distr = Categorical(logits=step_logits)
            entropy[i] = distr.entropy()

            sampled_token, log_prob, one_hot = self._sample_from(distr)
            sequence[i] = sampled_token
            logits[i] = log_prob
            one_hot_sequence[i] = one_hot

            prev_token_embedded = self.embed(one_hot)

        # to consistant with EGG implementation in loss function
        outputs = SpeakerOutputs(
            sequence.permute(1, 0),
            logits.permute(1, 0),
            entropy.permute(1, 0),
            one_hot_sequence.permute(1, 0, 2),
        )

        return one_hot_sequence, outputs

    def _sample_from(
        self, distr: Categorical
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        if self.training:
            if self.gumbel_softmax:
                one_hot = F.gumbel_softmax(distr.logits, hard=self.straight_through)
                token = torch.argmax(one_hot, dim=-1)
                prob = F.softmax(distr.logits, dim=-1) * one_hot
                log_prob = torch.log(prob.sum(dim=-1) + 1e-10)
                return token, log_prob, one_hot
            else:
                token = distr.sample()
                log_prob = distr.log_prob(token)
                one_hot = F.one_hot(token, num_classes=self.vocab_size).float()
                return token, log_prob, one_hot
        else:
            token = torch.argmax(distr.logits, dim=-1)
            log_prob = distr.log_prob(token)
            one_hot = F.one_hot(token, num_classes=self.vocab_size).float()
            return token, log_prob, one_hot


class TransformerListener(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
    ):
        super(TransformerListener, self).__init__()
        self.max_len = max_len
        self.token_embedding = nn.Linear(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_len + 1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, message: torch.FloatTensor) -> torch.Tensor:
        batch_size = message.size(1)

        token_embedded = self.token_embedding(message)
        token_embedded += self.pos_embedding.expand(-1, batch_size, -1)
        token_embedded = torch.cat(
            [self.cls_token_embedding.expand(-1, batch_size, -1), token_embedded],
            dim=0,
        )

        padding_mask = self._padding_mask(message)
        x = self.transformer(token_embedded, src_key_padding_mask=padding_mask)
        return x[0]

    def _padding_mask(self, message: torch.LongTensor) -> torch.BoolTensor:
        """create padding mask

        Args:
            message (torch.LongTensor): (max_len, batch_size)

        Returns:
            torch.BoolTensor: padding mask (batch_size, max_len + 2)
        """
        # from one-hot to token
        message = message.argmax(dim=-1).permute(1, 0)
        len = 1 + find_lengths(message)
        max_len = self.max_len
        batch_size = len.size(0)

        # for cls and eos token
        arange = torch.arange(max_len + 2, device=len.device)
        arange = arange.unsqueeze(0).expand(batch_size, -1)
        mask = arange >= len.unsqueeze(1)
        return mask


class TransformerSpeakerListener(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        max_len: int,
        speaker_dim: int,
        listener_dim: int,
        vocab_size: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        gumbe_softmax: bool,
        straight_through: bool,
        detach_message: bool,
        freeze: bool,
        listener_agent: nn.Module = None,
    ):
        super(TransformerSpeakerListener, self).__init__()
        self.detach_message = detach_message
        self.backbone = BackboneWrapper(
            backbone,
            output_size=speaker_dim,
            freeze=freeze,
        )
        self.speaker = TransformerSpeaker(
            max_len=max_len,
            d_model=speaker_dim,
            vocab_size=vocab_size,
            nhead=nhead,
            dropout=dropout,
            num_layers=num_layers,
            gumbel_softmax=gumbe_softmax,
            straight_through=straight_through,
            image_feature_size=(
                backbone.image_feature_size if backbone is Backbone else 1
            ),
        )
        self.listener = TransformerListener(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=listener_dim,
            nhead=nhead,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.listener_agent = listener_agent

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, SpeakerOutputs]:
        image_features = self.backbone(image)
        message, speaker_outputs = self.speaker(image_features)
        if self.detach_message:
            message = message.detach()
        x = self.listener(message)
        result = self.listener_agent(x)
        return result, speaker_outputs
