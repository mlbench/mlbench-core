import itertools

import torch
import torch.nn as nn

import mlbench_core.dataset.nlp.pytorch.wmt16.wmt16_config as config
from mlbench_core.models.pytorch.gnmt.attention import BahdanauAttention
from mlbench_core.models.pytorch.gnmt.utils import init_lstm_


class RecurrentAttention(nn.Module):
    """
    LSTM wrapped with an attention module.

    Args:
        input_size (int): number of features in input tensor
        context_size (int): number of features in output from encoder
        hidden_size (int): internal hidden size
        num_layers (int): number of layers in LSTM
        dropout (float): probability of dropout (on input to LSTM layer)
        init_weight (float): range for the uniform initializer
    """

    def __init__(
        self,
        input_size=1024,
        context_size=1024,
        hidden_size=1024,
        num_layers=1,
        dropout=0.2,
        init_weight=0.1,
        fusion=True,
    ):
        super(RecurrentAttention, self).__init__()

        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True, batch_first=False
        )
        init_lstm_(self.rnn, init_weight)

        self.attn = BahdanauAttention(
            hidden_size, context_size, context_size, normalize=True, fusion=fusion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_len):
        """
        Execute RecurrentAttention.

        Args:
            inputs (int): tensor with inputs
            hidden (int): hidden state for LSTM layer
            context: context tensor from encoder
            context_len: vector of encoder sequence lengths

        Returns:
            (rnn_outputs, hidden, attn_output, attn_scores)
        """
        # set attention mask, sequences have different lengths, this mask
        # allows to include only valid elements of context in attention's
        # softmax
        self.attn.set_mask(context_len, context)

        inputs = self.dropout(inputs)
        rnn_outputs, hidden = self.rnn(inputs, hidden)
        attn_outputs, scores = self.attn(rnn_outputs, context)

        return rnn_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):
    """
    Fully-connected classifier

    Args:
        in_features (int): number of input features
        out_features (int): number of output features (size of vocabulary)
        init_weight (float): range for the uniform initializer
    """

    def __init__(self, in_features, out_features, init_weight=0.1):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.

        Args:
            x (torch.tensor):

        Returns:
            torch.tensor
        """
        out = self.classifier(x)
        return out


class ResidualRecurrentDecoder(nn.Module):
    """
    Decoder with Embedding, LSTM layers, attention, residual connections and
    optinal dropout.

    Attention implemented in this module is different than the attention
    discussed in the GNMT arxiv paper. In this model the output from the first
    LSTM layer of the decoder goes into the attention module, then the
    re-weighted context is concatenated with inputs to all subsequent LSTM
    layers in the decoder at the current timestep.

    Residual connections are enabled after 3rd LSTM layer, dropout is applied
    on inputs to LSTM layers.

    Args:
        vocab_size (int): size of vocabulary
        hidden_size (int): hidden size for LSMT layers
        num_layers (int): number of LSTM layers
        dropout (float): probability of dropout (on input to LSTM layers)
        embedder (nn.Embedding): if None constructor will create new
            embedding layer
        init_weight (float): range for the uniform initializer
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=1024,
        num_layers=4,
        dropout=0.2,
        embedder=None,
        init_weight=0.1,
        fusion=True,
    ):
        super(ResidualRecurrentDecoder, self).__init__()

        self.num_layers = num_layers

        self.att_rnn = RecurrentAttention(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=1,
            dropout=dropout,
            fusion=fusion,
        )

        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(
                    2 * hidden_size,
                    hidden_size,
                    num_layers=1,
                    bias=True,
                    batch_first=False,
                )
            )

        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(
                vocab_size, hidden_size, padding_idx=config.PAD
            )
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)

        self.classifier = Classifier(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.
        Args:
            hidden: None or flattened hidden state for decoder RNN layers

        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(self.num_layers)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []
        return hidden

    def append_hidden(self, h):
        """
        Appends the hidden vector h to the list of internal hidden states.

        Args:
            h: hidden vector

        """
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        """
        Flattens the hidden state from all LSTM layers into one tensor (for
        the sequence generator).
        """
        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs, context, inference=False):
        """
        Execute the decoder.

        Args:
            inputs: tensor with inputs to the decoder
            context: state of encoder, encoder sequence lengths and hidden
                state of decoder's LSTM layers
            inference: if True stores and repackages hidden state

        Returns:

        """
        self.inference = inference

        enc_context, enc_len, hidden = context
        hidden = self.init_hidden(hidden)

        x = self.embedder(inputs)

        x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)
        self.append_hidden(h)

        x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.rnn_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.rnn_layers[i](x, hidden[i + 1])
            self.append_hidden(h)
            x = x + residual

        x = self.classifier(x)
        hidden = self.package_hidden()

        return x, scores, [enc_context, enc_len, hidden]
