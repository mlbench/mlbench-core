import torch
import torch.nn as nn
from torch.nn import functional as F

"""Following classes were taken and adapted from https://github.com/salesforce/awd-lstm-lm"""


class LockedDropout(nn.Module):
    """LockedDropout applies the same dropout mask to every time step.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(
            1 - self.p
        )
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + "p=" + str(self.p) + ")"


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """Applies a mask dropout to the embedding layer

    Args:
        embed (:obj:`torch.nn.Embedding`): Embedding layer to use
        words (:obj:`torch.Tensor`): Word inputs (tokenized)
        dropout (float): Dropout rate (Default 0.1)
        scale (float, optional): Scale factor for embedding weights

    Returns:
        (:obj:`torch.Tensor`) Output of Embedding after applying dropout mask to weights
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout
        ).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )
    return X


class WeightDrop(torch.nn.Module):
    """Weight Dropout layer. Wraps another module and patches the forward method to apply dropout to module weights.

    Args:
        module (:obj:`torch.nn.Module`): Module to wrap
        weights (listr[str]): Weights to apply dropout to
        dropout (float): Dropout rate (Default 0)

    """

    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        """Sets up new weights for the module"""
        for name_w in self.weights:
            print("Applying weight drop of {} to {}".format(self.dropout, name_w))
            # Make space for new weights
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            # Register raw weights
            self.module.register_parameter(name_w + "_raw", nn.Parameter(w.data))

    def _setweights(self):
        """Sets dropped out weights"""
        for name_w in self.weights:
            # Get raw weights and apply dropout
            raw_w = getattr(self.module, name_w + "_raw")
            w = F.dropout(raw_w, p=self.dropout, training=self.training)

            # This is because we may call this function in non-training mode first and so, as self.training=False, w is
            # a nn.Parameter and thus self.module.weight remains a Parameter of self.module when we don't want it to.
            if name_w in self.module._parameters:
                del self.module._parameters[name_w]
            # Set dropped out weights
            setattr(self.module, name_w, w)

    def forward(self, *args):
        """Forward patch"""
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)
