#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""
### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
  """Simple implementation of a Highway module, as described in the comment handouts.comment
    - Computes a projection of the input
    - Computes a combination projection from the input
    - Returns a combination of the projection and the original input
  """

  def __init__(self, embed_size, dropout_rate):
    """Init the Highway model. 

      @param embed_size (int): Embedding size (dimesionality). The input should be of size
        (batch_size, embed_size)
      @param dropout_rate (float): Dropout probability, for network.
    """
    super(Highway, self).__init__()
    self.embed_size = embed_size
    self.dropout_rate = dropout_rate

    self.projection = nn.Linear(
        in_features=embed_size, out_features=embed_size, bias=True)
    self.gate = nn.Linear(
        in_features=embed_size, out_features=embed_size, bias=True)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward(self, x):
    """ Run the model forward.

      @param x (Tensor): input tensor of token (batch_size, embed_size)
      @returns (Tensor): tensor of token (batch_size, embed_size)
    """
    projection = F.relu(self.projection(x))
    gate = torch.sigmoid(self.gate(x))
    highway = gate * projection + (1 - gate) * x
    embedding = self.dropout(highway)
    return embedding


### END YOUR CODE
