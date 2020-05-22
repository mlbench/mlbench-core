from torch import nn

from mlbench_core.models.pytorch.transformer.decoder import TransformerDecoder
from mlbench_core.models.pytorch.transformer.encoder import TransformerEncoder
from mlbench_core.models.pytorch.transformer.modules import build_embedding

DEFAULT_MAX_SOURCE_POSITIONS = 256
DEFAULT_MAX_TARGET_POSITIONS = 256


class TransformerModel(nn.Module):
    """Transformer model

    This model uses MultiHeadAttention as described in
    :cite:`NIPS2017_7181`

    Args:
        args: Arguments of model. All arguments should be accessible via `__getattribute__` method
        src_dict (:obj:`mlbench_core.dataset.nlp.pytorch.wmt17.Dictionary`): Source dictionary
        trg_dict (:obj:`mlbench_core.dataset.nlp.pytorch.wmt17.Dictionary`): Target dictionary
    """

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

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
    ):
        """
        Run the forward pass of the transformer model.

        Args:
            src_tokens (:obj:`torch.Tensor`): Source tokens
            src_lengths (:obj:`torch.Tensor`): Source sentence lengths
            prev_output_tokens (:obj:`torch.Tensor`): Previous output tokens

        Returns:
            (:obj:`torch.Tensor`, Optional[:obj:`torch.Tensor`]):
                The model output, and attention weights if needed
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions(), self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder.

        Returns:
            (int)
        """
        return self.decoder.max_positions()
