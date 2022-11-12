__author__ = 'Alaie Titor'
__all__ = ['MLP']

import copy
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Dropout, LayerNorm, Linear, ModuleList, ReLU, Sequential)


class MLP(Sequential):
    def __init__(self, channels: List[int], bias: bool = True, dropout: float = 0., norm: str = 'batch'):

        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                if norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)
