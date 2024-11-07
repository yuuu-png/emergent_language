from typing import Optional

from torch import Tensor
import torch.nn as nn


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # ::Attention:: Just only modified here to get the attention weights in hook function
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=True,
        )[0]
        return self.dropout2(x)
