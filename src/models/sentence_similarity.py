import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core import find_lengths

from .backbone.backbone import BackboneWrapper, Backbone
from src.models.transformer_agents import TransformerSpeaker, SpeakerOutputs


class InstanceIdentifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        speaker: TransformerSpeaker,
        d_model: int,
        larger_max_len: int,
        smaller_max_len: int,
        detach_message: bool,
        vocab_size: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        only_positive_samples: bool,
    ):
        super(InstanceIdentifier, self).__init__()
        self.detach_message = detach_message
        self.backbone = backbone
        self.listener = SentenceSimilarity(
            vocab_size=vocab_size,
            larger_max_len=larger_max_len,
            smaller_max_len=smaller_max_len,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.speaker = speaker
        self.smaller_max_len = smaller_max_len
        self.only_positive_samples = only_positive_samples
        self.fc = nn.Linear(d_model, 1)

    def forward(
        self,
        larger_image: torch.Tensor,
        smaller_image: torch.Tensor,
        smallers_len: list[int],
    ) -> tuple[torch.Tensor, SpeakerOutputs, SpeakerOutputs]:
        larger_image_futures = self.backbone(larger_image)
        smaller_image_futures = self.backbone(smaller_image)

        larger_message, larger_speaker_outputs = self.speaker(larger_image_futures)
        smaller_message, smaller_speaker_outputs = self.speaker(smaller_image_futures)
        smaller_message = smaller_message[: self.smaller_max_len + 1]

        if self.detach_message:
            larger_message = larger_message.detach()
            smaller_message = smaller_message.detach()

        if self.only_positive_samples:
            loss = self.loss_with_only_positive_sample(
                larger_message, smaller_message, smallers_len
            )
        else:
            loss = self.loss_calulated_by_series(
                larger_message, smaller_message, smallers_len
            )

        return loss, larger_speaker_outputs, smaller_speaker_outputs

    def loss_with_only_positive_sample(
        self,
        larger_message: torch.Tensor,
        smaller_message: torch.Tensor,
        smallers_len: list[int],
    ):
        num_larger = larger_message.size(1)
        num_smaller = smaller_message.size(1)

        duplicated_larger_message = torch.zeros(
            (larger_message.size(0), num_smaller, larger_message.size(2)),
            device=larger_message.device,
        )
        current = 0
        for i, smaller_len in enumerate(smallers_len):
            duplicated_larger_message[:, current : current + smaller_len] = (
                larger_message[:, i : i + 1]
            )
            current += smaller_len
        similarity = self.listener(duplicated_larger_message, smaller_message)
        similarity = self.fc(similarity).squeeze(1)
        label = torch.ones(
            num_smaller, device=smaller_message.device, dtype=torch.float
        )
        loss = F.cross_entropy(similarity, label)
        return loss

    def loss_calulated_by_series(
        self,
        larger_message: torch.Tensor,
        smaller_message: torch.Tensor,
        smallers_len: list[int],
    ):
        num_larger = larger_message.size(1)
        num_smaller = smaller_message.size(1)

        total_loss = torch.tensor(0.0, device=larger_message.device)
        current = 0
        for i, smaller_len in enumerate(smallers_len):
            label = torch.zeros(
                num_smaller, device=smaller_message.device, dtype=torch.float
            )
            label[current : current + smaller_len] = 1

            duplicated_larger_message = larger_message[:, i : i + 1].repeat_interleave(
                num_smaller, dim=1
            )
            similarity = self.listener(duplicated_larger_message, smaller_message)
            similarity = self.fc(similarity).squeeze(1)
            total_loss += F.cross_entropy(similarity, label)

            current += smaller_len
        loss = total_loss / len(smallers_len)
        return loss

    def loss_calculated_by_parallel(
        self,
        larger_message: torch.Tensor,
        smaller_message: torch.Tensor,
        smallers_len: list[int],
    ):
        """メモリが足りなくなる"""
        num_larger = larger_message.size(1)
        num_smaller = smaller_message.size(1)

        duplicated_larger_message = larger_message.repeat_interleave(num_smaller, dim=1)
        duplicated_smaller_message = smaller_message.repeat(1, num_larger, 1)

        similarity = self.listener(
            duplicated_larger_message, duplicated_smaller_message
        )

        # create the label from the length of the smallers including the larger
        label = torch.zeros(
            (num_smaller * num_larger), device=smaller_message.device, dtype=torch.long
        )
        current = 0
        for i, l in enumerate(smallers_len):
            label[i * num_smaller + current : i * num_smaller + current + l] = 1
            current += l

        loss = F.cross_entropy(similarity, label)
        return loss


class SentenceSimilarity(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        larger_max_len: int,
        smaller_max_len: int,
        d_model: int,
        nhead: int,
        dropout: float,
        num_layers: int,
    ):
        super(SentenceSimilarity, self).__init__()
        self.larger_max_len = larger_max_len
        self.smaller_max_len = smaller_max_len
        self.max_len = 1 + larger_max_len + 1 + smaller_max_len

        self.token_embedding = nn.Linear(vocab_size, d_model)
        self.larger_pos_embedding = nn.Parameter(
            torch.randn(larger_max_len + 1, 1, d_model)
        )
        self.smaller_pos_embedding = nn.Parameter(
            torch.randn(smaller_max_len + 1, 1, d_model)
        )
        self.sentence_embedding = nn.Parameter(torch.randn(2, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token_embedding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, larger_message: torch.Tensor, smaller_message: torch.Tensor):
        N = larger_message.size(1)

        larger_token_embedded = self.token_embedding(larger_message)
        larger_token_embedded += self.larger_pos_embedding.expand(-1, N, -1)
        smaller_token_embedded = self.token_embedding(smaller_message)
        smaller_token_embedded += self.smaller_pos_embedding.expand(-1, N, -1)
        token_embedded = torch.cat(
            [
                self.cls_token_embedding.expand(-1, N, -1),
                larger_token_embedded,
                self.sep_token_embedding.expand(-1, N, -1),
                smaller_token_embedded,
            ],
            dim=0,
        )

        larger_message_mask = self._padding_mask(
            larger_message, max_len=self.larger_max_len
        )
        smaller_message_mask = self._padding_mask(
            smaller_message, max_len=self.smaller_max_len
        )
        padding_mask = torch.concat((larger_message_mask, smaller_message_mask), dim=1)
        x = self.transformer(token_embedded, src_key_padding_mask=padding_mask)
        return x[0]

    def _padding_mask(
        self, message: torch.LongTensor, max_len: int
    ) -> torch.BoolTensor:
        """create padding mask

        Args:
            message (torch.LongTensor): (max_len, batch_size)

        Returns:
            torch.BoolTensor: padding mask (batch_size, max_len + 2)
        """
        # from one-hot to token
        message = message.argmax(dim=-1).permute(1, 0)
        len = 1 + find_lengths(message)
        batch_size = len.size(0)

        # for cls and eos token
        arange = torch.arange(max_len + 2, device=len.device)
        arange = arange.unsqueeze(0).expand(batch_size, -1)
        mask = arange >= len.unsqueeze(1)
        return mask
