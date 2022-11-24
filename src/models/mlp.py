__author__ = 'Alaie Titor'
__all__ = ['MLP']

from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import  GraphNorm, LayerNorm, PairNorm, InstanceNorm, MessageNorm


class MLP(torch.nn.Module):
    def __init__(self, channels: List[int], bias: bool = True, norm: str = 'graph', eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.lins = ModuleList()
        self.norms = ModuleList() if norm is not None else None

        for i in range(1, len(channels)):
            self.lins.append(Linear(channels[i - 1], channels[i], bias=bias))

            if norm and i < len(channels) - 1:
                if norm == 'graph':
                    self.norms.append(GraphNorm(channels[i]))
                if norm == 'layer':
                    self.norms.append(LayerNorm(channels[i], affine=True))
                if norm == 'instance':
                    self.norms.append(InstanceNorm(channels[i], affine=True))
                if norm == 'pair':
                    self.norms.append(PairNorm(scale=1, scale_individually=True))


        self.num_layers = len(self.lins)

    def forward(self, x: Tensor, *args, **kwargs):
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)

            if self.norms is not None:
                x = self.norms[i](x, *args, **kwargs)
                
            x = F.relu(x) + self.eps

        x = self.lins[-1](x)

        return x