import math

import torch
import torch.nn.functional as F
from torch import nn

from mlbench_core.models.pytorch.transformer.modules import (
    PositionalEmbedding,
    TransformerEncoderLayer,
)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args: Arguments of model. All arguments should be accessible via `__getattribute__` method
        dictionary (:obj:`mlbench_core.dataset.nlp.pytorch.wmt17.Dictionary`): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool): Pad sources to the left (`True`) or right (`False`). Default: `True`
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.softmax_type = args.softmax_type

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                left_pad=left_pad,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, src_tokens):
        """Forward function of encoder

        Args:
            src_tokens (:obj:`torch.Tensor`): Source tokens

        Returns:
            (dict): {`encoder:out` (:obj:`torch.Tensor`), `encoder_padding_mask` (:obj:`torch.Tensor`)}
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if x.size(1) == 1:
            if x.is_contiguous():
                x = x.view(x.size(0), x.size(1), x.size(2))
            else:
                x = x.contiguous()
        else:
            x = x.contiguous()

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        if (self.softmax_type == "fast_fill") and (encoder_padding_mask is not None):
            encoder_padding_mask = torch.zeros_like(
                encoder_padding_mask, dtype=x.dtype
            ).masked_fill_(encoder_padding_mask, float("-inf"))

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())
