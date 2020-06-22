import torch.nn as nn
from torch.nn.functional import log_softmax

from mlbench_core.dataset.nlp.pytorch.wmt16 import wmt16_config
from mlbench_core.models.pytorch.gnmt.decoder import ResidualRecurrentDecoder
from mlbench_core.models.pytorch.gnmt.encoder import ResidualRecurrentEncoder


class Seq2Seq(nn.Module):
    """
    Generic Seq2Seq module, with an encoder and a decoder.
    Args:
        encoder (Encoder): Model encoder
        decoder (Decoder): Model decoder
    """

    def __init__(self, encoder=None, decoder=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, inputs, lengths):
        """
        Applies the encoder to inputs with a given input sequence lengths.

        Args:
            inputs (torch.tensor): tensor with inputs (seq_len, batch)
            lengths: vector with sequence lengths (excluding padding)

        Returns:
            torch.tensor
        """
        return self.encoder(inputs, lengths)

    def decode(self, inputs, context, inference=False):
        """
        Applies the decoder to inputs, given the context from the encoder.

        Args:
            inputs (torch.tensor): tensor with inputs (seq_len, batch)
            context: context from the encoder
            inference: if True inference mode, if False training mode

        Returns:
            torch.tensor
        """
        return self.decoder(inputs, context, inference)

    def generate(self, inputs, context, beam_size):
        """
        Autoregressive generator, works with SequenceGenerator class.
        Executes decoder (in inference mode), applies log_softmax and topK for
        inference with beam search decoding.

        Args:
            inputs: tensor with inputs to the decoder
            context: context from the encoder
            beam_size: beam size for the generator

        Returns:
            (words, logprobs, scores, new_context)
            words: indices of topK tokens
            logprobs: log probabilities of topK tokens
            scores: scores from the attention module (for coverage penalty)
            new_context: new decoder context, includes new hidden states for
            decoder RNN cells
        """
        logits, scores, new_context = self.decode(inputs, context, True)
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context


class GNMT(Seq2Seq):
    """
    GNMT v2 model

    Args:
        vocab_size (int): size of vocabulary (number of tokens)
        hidden_size (int): internal hidden size of the model
        num_layers (int): number of layers, applies to both encoder and
            decoder
        dropout (float): probability of dropout (in encoder and decoder)
            tensors, if false the model uses (seq, batch, feature)
        share_embedding (bool): if True embeddings are shared between
            encoder and decoder
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=1024,
        num_layers=4,
        dropout=0.2,
        share_embedding=True,
        fusion=True,
    ):
        super(GNMT, self).__init__()

        if share_embedding:
            embedder = nn.Embedding(
                vocab_size, hidden_size, padding_idx=wmt16_config.PAD
            )
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(
            vocab_size, hidden_size, num_layers, dropout, embedder
        )

        self.decoder = ResidualRecurrentDecoder(
            vocab_size, hidden_size, num_layers, dropout, embedder, fusion=fusion
        )

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output
