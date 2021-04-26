# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """feed forward network in Transformer"""

    def __init__(self, d_model, dim_ffn, dropout=0.1):
        """

        Args:
            d_model (int): model dimension.
            dim_ffn (int): dim of internal manifold.

        Kwargs:
            dropout (TODO): TODO


        """
        nn.Module.__init__(self)
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.dim_ffn = dim_ffn
        self.dropout = dropout

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_ffn, d_model)

    def forward(self, x):
        # x: (bs, nsamples, d_model)
        src = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(self.dropout(x))
        return x + src


class PositionalEncoding(nn.Module):
    """encode the timestep as positional encoding"""

    def __init__(self, d_model, scale_factor=100):
        """
        Args:
            d_model (int): model dim.
            scale_factor (int): scale the time steps.

        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                (-math.log(10000.0) / d_model))    # ( d_model/2, )
        self.register_buffer('div_term', div_term)

    def forward(self, ts):
        # ts: (nsamples,)
        position = ts * self.scale_factor
        nsamples = position.size()[0]
        position = position.unsqueeze(1)    # (nsamples, 1)
        pe = torch.stack([torch.sin(position * self.div_term),
                            torch.cos(position * self.div_term)], dim=1)    # (nsamples, 2, d_model/2)
        return pe.reshape([1, nsamples,
                            self.d_model])    # (1, 1, d_model), corresponding to (bs, nsamples, d_model)


def create_local_attn_mask(nsamples, band_width=10):
    """create local attention mask.

    Args:
        nsamples (int): The number of samples.
        band_width (int): The width each sample can see. Centered by the sample.

    Returns: (nsamples, nsamples)

    """
    half_band = band_width // 2
    ones = torch.ones([nsamples, nsamples], dtype=torch.float32)
    band_up = torch.triu(ones, diagonal=half_band)
    band_lower = torch.tril(ones, diagonal=half_band)
    mask = band_up * band_lower
    return mask


class TransformerLayer(nn.Module):
    """Encoder for X: (bs, nsamples, nc)
    Output is the encoded features (bs, nsamples, nc_out) 

    """

    def __init__(self, nc_out, nc_in, nhidden, dropout=0.1):
        """
        Args:
            nc_out (int): output feature dimensions.
            nc_in (int): nc input.
            nhidden (int): hidden size.
            dropout (int): The dropout rate. Default 0.1.
            attn_bandwidth (int): the width of the attention. Default -1 means the width is the sequence length.
        """
        super(TransformerLayer, self).__init__()

        self.nc_out = nc_out
        self.nc_in = nc_in
        self.nhidden = nhidden

        self.i2h = nn.Linear(nc_in, nhidden)
        self.pe = PositionalEncoding(nhidden, scale_factor=100)

        self.self_attn = nn.MultiheadAttention(nhidden, 1, dropout=dropout)
        self.norm1 = nn.LayerNorm(nhidden)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForwardNetwork(nhidden, nhidden * 2, dropout=dropout)
        self.norm2 = nn.LayerNorm(nhidden)
        self.dropout2 = nn.Dropout(dropout)

        self.h2o = nn.Linear(nhidden, nc_out)

    def forward(self, x, ts, mask=None):
        #x: (bs, nsamples, nc_in)
        #ts: (nsamples,)
        # h = self.i2h(x) + self.pe(ts)    # (bs, nsamples, nhidden)
        h = self.i2h(x)    # no pe
        # h = self.i2h(x) * math.sqrt(self.nhidden) + self.pe(ts)    # multiply embedding

        h = h.permute(1, 0, 2)    # to: (nsamples, bs, nhidden)
        h2 = self.self_attn(h, h, h, attn_mask=mask)[0]
        h = h + self.dropout1(h2)
        h = h.permute(1, 0, 2)    # back: (bs, nsamples, nhidden)

        h = self.norm1(h)

        h2 = self.ffn(h)
        h = h + self.dropout2(h2)
        h = self.norm2(h)

        o = self.h2o(h)
        return o    # (bs, nsamples, nc_out)
