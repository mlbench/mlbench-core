import math

import torch
import torch.nn.functional as F
from torch import nn

from mlbench_core.models.pytorch.transformer.modules import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
)

# from apex.normalization.fused_layer_norm import FusedLayerNorm


# Copied from mlperf.
class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False
    ):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                left_pad=left_pad,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)
        self.normalize = args.decoder_normalize_before

        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)  # TODO change to FUsedLayerNorm

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None
                else None,
                incremental_state,
            )

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x, attn

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "decoder.embed_positions.weights" in state_dict:
                del state_dict["decoder.embed_positions.weights"]
            state_dict["decoder.embed_positions._float_tensor"] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "decoder.layers.{}.layer_norms.{}.{}".format(i, old, m)
                    if k in state_dict:
                        state_dict[
                            "decoder.layers.{}.{}.{}".format(i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        return state_dict

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, "reorder_incremental_state"):
                module.reorder_incremental_state(
                    incremental_state, new_order,
                )

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, "_beam_size", -1) != beam_size:

            def apply_set_beam_size(module):
                if module != self and hasattr(module, "set_beam_size"):
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1, dtype=torch.float32)
