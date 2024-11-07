from torch import Tensor
import torch.nn as nn
from src.models.custom_transformer import CustomTransformerDecoderLayer


class SaveCrossAttentionMapMixin:
    """This mixin class is used to save the attention map of the cross-attention block in the Transformer.
    You should use this class with the `src.models.custom_transformer.CustomTransformerDecoderLayer` class.

    Returns:
        _type_: _description_
    """

    @property
    def attention_map(self) -> list[Tensor]:
        # (batch size, target sequence length, source sequence length)
        # target: output sequence of Transformer (input and output of the decoder)
        # source: input sequence of Transformer (encoded output of the encoder)
        return self._attention_map

    def set_hook(self, module: nn.TransformerDecoder):
        for layer in module.layers:
            layer: CustomTransformerDecoderLayer
            layer.multihead_attn.register_forward_hook(self.save_attention_map_hook)

    def reset_attention_map(self):
        self._attention_map = []

    def save_attention_map_hook(self, module, input, output):
        self._attention_map.append(output[1])
