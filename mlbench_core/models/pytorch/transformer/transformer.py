import torch
import torch.nn.functional as F
from torch import nn

from mlbench_core.models.pytorch.transformer import utils
from mlbench_core.models.pytorch.transformer.decoder import TransformerDecoder
from mlbench_core.models.pytorch.transformer.encoder import TransformerEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 256
DEFAULT_MAX_TARGET_POSITIONS = 256


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()

    emb = utils.Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb


# copied from mlperf
class TransformerModel(nn.Module):
    """Transformer model TODO add link
    Args:
        args: Arguments of model
        src_dict: Source dictionary
        trg_dict: Target dictionary
    """

    # adapted from mlperf
    def __init__(self, args, src_dict, trg_dict):
        super().__init__()
        self._is_generation_fast = False
        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        # Define embedding layer
        if args.share_all_embeddings:
            if src_dict != trg_dict:
                raise ValueError("share_all_embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "share_all_embeddings requires encoder_embed_dim to match decoder_embed_dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "share_all_embeddings not compatible with decoder_embed_path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                trg_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        self.encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        self.decoder = TransformerDecoder(args, trg_dict, decoder_embed_tokens)

    # adapted from mlperf
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
    ):
        """
        Run the forward pass of the transformer model.

        Args:
            src_tokens (`obj`:torch.Tensor): Source tokens
            src_lengths (`obj`:torch.Tensor): Source sentence lengths
            prev_output_tokens (`obj`:torch.Tensor): Previous output tokens

        Returns:
            TODO add decoder output
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    # copied from mlperf
    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions(), self.decoder.max_positions()

    # copied from mlperf
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

    # copied from mlperf
    def max_decoder_positions(self):
        """Maximum length supported by the decoder.

        Returns:
            (int)
        """
        return self.decoder.max_positions()

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, "make_generation_fast_"):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        # this model should no longer be used for training
        self.eval()
        self.train = train
