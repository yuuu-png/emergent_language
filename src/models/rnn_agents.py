import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from egg.core import RnnReceiverDeterministic, find_lengths

from .backbone.backbone import BackboneWrapper
from .pretrain_vision import Vision


class SpeakerOutputs:
    def __init__(self, sequence, logits, entropy):
        self.sequence = sequence
        self.logits = logits
        self.entropy = entropy


class RnnSpeaker(nn.Module):
    def __init__(
        self,
        backbone: BackboneWrapper,
        max_len: int,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
    ):
        super(RnnSpeaker, self).__init__()
        self.backbone = backbone
        self.max_len = max_len
        self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_dim)

        self.bos_emb = nn.Parameter(torch.randn(1, embed_dim))
        self.embed = nn.Embedding(embedding_dim=embed_dim, num_embeddings=vocab_size)
        self.embed_fc = nn.Linear(embed_dim, vocab_size)
        self.logits = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, SpeakerOutputs]:
        batch_size = x.size(0)

        hidden = self.backbone(x)

        sequence = []
        logits = []
        entropy = []

        prev_token_embedded = self.bos_emb.expand(batch_size, -1)
        for _ in range(self.max_len):
            hidden = self.cell(prev_token_embedded, hidden)
            step_logits = F.log_softmax(self.logits(hidden), dim=-1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            x = self._sample(distr)
            logits.append(distr.log_prob(x))

            prev_token_embedded = self.embed(x)
            sequence.append(x)

        sequence = torch.stack(sequence, dim=1)
        logits = torch.stack(logits, dim=1)
        entropy = torch.stack(entropy, dim=1)
        outputs = SpeakerOutputs(sequence, logits, entropy)

        return sequence, outputs

    def _sample(self, distr: Categorical) -> torch.Tensor:
        if self.training:
            return distr.sample()
        else:
            return torch.argmax(distr.logits, dim=-1)


class RnnListener(nn.Module):
    class Backbone(nn.Module):
        def __init__(self, input_size, output_size=784):
            super(RnnListener.Backbone, self).__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, channel_input, receiver_input=None, aux_input=None):
            x = self.fc(channel_input)
            return torch.sigmoid(x)

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        embed_dim,
        agent: nn.Module = None,
    ):
        if agent is None:
            agent = RnnListener.Backbone(input_size=hidden_dim)
        super(RnnListener, self).__init__()
        self.rnn = RnnReceiverDeterministic(
            agent=agent,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_dim,
            cell="gru",
        )

    def forward(self, x: torch.Tensor, len: torch.Tensor) -> torch.Tensor:
        x, _, _ = self.rnn(message=x, lengths=len)
        return x


from egg.core import RnnSenderReinforce


class Sender(nn.Module):
    def __init__(self, backbone, output_size):
        super(Sender, self).__init__()
        self.fc = nn.Linear(500, output_size)
        self.backbone = backbone

    def forward(self, x, aux_input=None):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.fc(x)
        return x


class RnnSpeakerListener(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        max_len: int,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        listener_agent: nn.Module = None,
    ):
        backbone_output_size = 500 if backbone is Vision else 2048
        super(RnnSpeakerListener, self).__init__()
        self.speaker = RnnSpeaker(
            backbone=BackboneWrapper(
                backbone,
                backbone_output_size=backbone_output_size,
                output_size=hidden_dim,
            ),
            max_len=max_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )
        self.listener = RnnListener(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            agent=listener_agent,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, SpeakerOutputs]:
        message, speaker_outputs = self.speaker(x)
        message_length = find_lengths(message)
        reconstruction = self.listener(message, message_length)
        return reconstruction, speaker_outputs
