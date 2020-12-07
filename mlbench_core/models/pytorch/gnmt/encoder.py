import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import mlbench_core.dataset.nlp.pytorch.wmt16.wmt16_config as config
from mlbench_core.models.pytorch.gnmt.utils import init_lstm_


class ResidualRecurrentEncoder(nn.Module):
    """
    Encoder with Embedding, LSTM layers, residual connections and optional
    dropout.

    The first LSTM layer is bidirectional and uses variable sequence length
    API, the remaining (num_layers-1) layers are unidirectional. Residual
    connections are enabled after third LSTM layer, dropout is applied on
    inputs to LSTM layers.

    Args:
        vocab_size: size of vocabulary
        hidden_size: hidden size for LSTM layers
        num_layers: number of LSTM layers, 1st layer is bidirectional
        dropout: probability of dropout (on input to LSTM layers)
        embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        init_weight: range for the uniform initializer
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=1024,
        num_layers=4,
        dropout=0.2,
        embedder=None,
        init_weight=0.1,
    ):
        super(ResidualRecurrentEncoder, self).__init__()
        self.rnn_layers = nn.ModuleList()
        # 1st LSTM layer, bidirectional
        self.rnn_layers.append(
            nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                bias=True,
                batch_first=False,
                bidirectional=True,
            )
        )

        # 2nd LSTM layer, with 2x larger input_size
        self.rnn_layers.append(
            nn.LSTM(
                (2 * hidden_size),
                hidden_size,
                num_layers=1,
                bias=True,
                batch_first=False,
            )
        )

        # Remaining LSTM layers
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(
                    hidden_size,
                    hidden_size,
                    num_layers=1,
                    bias=True,
                    batch_first=False,
                )
            )

        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        self.dropout = nn.Dropout(p=dropout)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(
                vocab_size, hidden_size, padding_idx=config.PAD
            )
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)

    def forward(self, inputs, lengths):
        """
        Execute the encoder.

        Args:
            inputs: tensor with indices from the vocabulary
            lengths: vector with sequence lengths (excluding padding)

        Returns:
            tensor with encoded sequences

        """
        x = self.embedder(inputs)

        # bidirectional layer
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=False)
        x, _ = self.rnn_layers[0](x)
        x, _ = pad_packed_sequence(x, batch_first=False)

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = x + residual

        return x
