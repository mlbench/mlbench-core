import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mlbench_core.models.pytorch.gnmt import attn_score


class AttentionScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att_query, att_keys, bias, linear_att):
        score = attn_score.forward(att_query, att_keys, bias, linear_att)
        ctx.save_for_backward(att_query, att_keys, bias, linear_att)
        return score

    @staticmethod
    def backward(ctx, grad_output):
        att_query, att_keys, bias, linear_att = ctx.saved_tensors
        grad_query, grad_keys, grad_bias, grad_linear_att = attn_score.backward(
            grad_output, att_query, att_keys, bias, linear_att
        )
        return grad_query, grad_keys, grad_bias, grad_linear_att


fused_calc_score = AttentionScore.apply


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to tf.contrib.seq2seq.BahdanauAttention

    Args:
        query_size (int): feature dimension for query
        key_size (int): feature dimension for keys
        num_units (int): internal feature dimension
        normalize (bool): whether to normalize energy term.
            Default: `False`
        init_weight (float): range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector.
            Default: 0.1
    """

    def __init__(
        self,
        query_size,
        key_size,
        num_units,
        normalize=False,
        init_weight=0.1,
        fusion=True,
    ):
        super(BahdanauAttention, self).__init__()

        self.normalize = normalize
        self.num_units = num_units

        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = Parameter(torch.Tensor(num_units))

        self.mask = None

        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter("normalize_scalar", None)
            self.register_parameter("normalize_bias", None)

        self.fusion = fusion
        self.reset_parameters(init_weight)

    def reset_parameters(self, init_weight):
        """
        Sets initial random values for trainable parameters.

        Args:
            init_weight (float):
        """
        stdv = 1.0 / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        Sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        Args:
            context_len (`obj`:torch.Tensor):
            context (`obj`:torch.Tensor): (t_k x b x n)

        Returns:

        """
        max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64, device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score

        Args:
            att_query (`obj`:torch.Tensor):
            att_keys (`obj`:torch.Tensor):

        Returns:

        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys

        if self.normalize:
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def forward(self, query, keys):
        """

        Args:
            query (`obj`:torch.Tensor): (t_q x b x n)
            keys (`obj`:torch.Tensor): (t_k x b x n)

        Returns:
            (context, scores_normalized)
        context: (t_q x b x n)
        scores_normalized: (t_q x b x t_k)

        """

        # first dim of keys and query has to be 'batch', it's needed for bmm
        keys = keys.transpose(0, 1)
        if query.dim() == 3:
            query = query.transpose(0, 1)

        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)

        # FC layers to transform query and key
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)

        # scores: (b x t_q x t_k)
        if self.fusion:
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
            scores = fused_calc_score(
                processed_query, processed_key, self.normalize_bias, linear_att
            )
        else:
            scores = self.calc_score(processed_query, processed_key)

        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            # I can't use -INF because of overflow check in pytorch
            scores.data.masked_fill_(mask, -65504.0)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (b x t_q x n)
        context = torch.bmm(scores_normalized, keys)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        else:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)

        return context, scores_normalized
