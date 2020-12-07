import torch
import torch.nn as nn

from mlbench_core.models.pytorch.layers import (
    LockedDropout,
    WeightDrop,
    embedded_dropout,
)


class LSTMLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.

    Model taken from https://github.com/salesforce/awd-lstm-lm

    Args:
        ntoken (int): Number of tokens in vocabulary
        ninp (int): Embedding size (LSTM input size)
        nhid (int): Number of hidden LSTM units per layer
        nlayers (int): Number of LSTM layers
        dropout (float): Output dropout rate (LockedDropout). Default 0.5
        dropouth (float): LSTM output dropout rate (between each layer except for last). Default 0.5
        dropouti (float): Input dropout to LSTM layers. Default 0.5
        dropoute (float): Embedding dropout. Default 0.1
        wdrop (float): Weight dropout for LSTM layers. Default 0
        tie_weights (bool): If True, encoder and decoder weights are tied. Default False

    """

    def __init__(
        self,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
    ):
        super(LSTMLanguageModel, self).__init__()
        self.lockdroph = LockedDropout(p=dropouth)
        self.lockdropi = LockedDropout(p=dropouti)
        self.lockdrop = LockedDropout(p=dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [
            torch.nn.LSTM(
                ninp if l == 0 else nhid,
                nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                1,
                dropout=0,
            )
            for l in range(nlayers)
        ]
        if wdrop:
            self.rnns = [
                WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop) for rnn in self.rnns
            ]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        # Embedded Dropout
        emb = embedded_dropout(
            self.encoder, input, dropout=self.dropoute if self.training else 0
        )
        # LSTM input dropout
        emb = self.lockdropi(emb)

        # Manual feeding of LSTM layers
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        # Iterate on all LSTM layers
        for l, rnn in enumerate(self.rnns):
            # Compute output and hidden state
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            # Apply LockDrop if not last layer
            if l != self.nlayers - 1:
                raw_output = self.lockdroph(raw_output)
                outputs.append(raw_output)
        hidden = new_hidden

        # Output dropout
        output = self.lockdrop(raw_output)
        outputs.append(output)
        #
        result = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [
            (
                weight.new(
                    1,
                    bsz,
                    self.nhid
                    if l != self.nlayers - 1
                    else (self.ninp if self.tie_weights else self.nhid),
                ).zero_(),
                weight.new(
                    1,
                    bsz,
                    self.nhid
                    if l != self.nlayers - 1
                    else (self.ninp if self.tie_weights else self.nhid),
                ).zero_(),
            )
            for l in range(self.nlayers)
        ]
