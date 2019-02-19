#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  """Simple implementation of a CNN module, as described in the homework handout.
    - Takes as input a batch of word (a word is a vector of character embeddings) of dimension
      (batch_size x char_embed_size x max_word_length)
  """

  def __init__(self,
               char_embed_size,
               word_embed_size,
               window_size=5,
               max_word_length=21):
    """Init the Highway model. 

      @param char_embed_size (int): Embedding size (dimesionality) of the characters.
        The input should be of size (batch_size, char_embed_size, max_word_length)
      @param word_embed_size (int): Target embedding size (dimensionality) for each word.
        The output will be of size (batch_size, word_embed_size)
      @param window_size (int): The kernel size for the window along the character dimension (dim = 2)
      @param max_word_length (int): The length of the maximum word in this batch.
    """
    super(CNN, self).__init__()
    self.char_embed_size = char_embed_size
    self.word_embed_size = word_embed_size
    self.window_size = window_size
    self.max_word_length = max_word_length

    self.conv1d = nn.Conv1d(
        in_channels=char_embed_size,
        out_channels=word_embed_size,
        kernel_size=window_size,
        bias=True)
    # This is the third dimension of the conv1d output from above.
    # (batch_size x word_embed_size x out_window_size)
    out_window_size = max_word_length - window_size + 1
    self.max_pool = nn.MaxPool1d(kernel_size=out_window_size)

  def forward(self, x):
    """ Run the model forward.

      @param x (Tensor): intput tensor of token (batch_size, char_embed_size, max_word_length)
      @returns (Tensonr): output tensor of token (batch_size, word_embed_size)
    """
    conv = F.relu(self.conv1d(x))
    conv_out = self.max_pool(conv)
    return conv_out


### END YOUR CODE
