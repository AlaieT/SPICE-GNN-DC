__author__ = 'Alaie Titor'
__all__ = ['Extractor', 'GGAConv', 'Decoder']

import copy
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, ModuleList
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessageNorm, MessagePassing, DeepGCNLayer
from typing import List
from .mlp import MLP


class GGAConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            aggr: str = 'softmax',
            learn_t: bool = True,
            t: float = 1.0,
            num_layers: int = 2,
            dropout: float = 0.0,
            eps: float = 1e-7):

        aggr_kwargs = {}
        if aggr == 'softmax':
            aggr_kwargs = dict(t=t, learn=learn_t)

        super().__init__(aggr=aggr, aggr_kwargs=aggr_kwargs)

        channels = [in_channels] + [hidden_channels * 2 for _ in range(num_layers - 1)] + [out_channels]

        self.eps = eps
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.msg_norm = MessageNorm(True)
        self.mlp = MLP(channels=channels, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = (x, x)
        out = self.propagate(x=x, edge_index=edge_index)
        out = self.msg_norm(x[0], out)
        out += x[1]

        return self.mlp(out)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        msg = x_j
        return msg.relu() + self.eps


class DeepGGALayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, ckpt_grad: bool = False, block_size: int = 2, dropout: float = 0.0, eps=1e-7) -> None:
        super().__init__()

        self.dropout = dropout
        self.eps = eps
        self.ckpt_grad = ckpt_grad
        self.block_size = block_size

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(block_size):
            self.convs.append(GGAConv(hidden_channels, hidden_channels, hidden_channels))
            self.norms.append(BatchNorm1d(hidden_channels, affine=True))

    def forward(self, *args, **kwargs) -> Tensor:
        args = list(args)
        x: Tensor = args.pop(0)

        intter = x

        for i in range(self.block_size):
            if self.ckpt_grad and x.requires_grad:
                x = checkpoint(self.convs[i], x, *args, **kwargs)
            else:
                x = self.convs[i](x, *args, **kwargs)

            x = self.norms[i](x)

            if i < self.block_size - 1:
                x = F.relu(x) + self.eps

        x = F.relu(intter + x) + self.eps
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class Extractor(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.layers = ModuleList()

        for i in range(1, self.num_layers + 1):
            layer = DeepGGALayer(hidden_channels, block_size=2, ckpt_grad=i % 3, dropout=dropout)
            self.layers.append(layer)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_heads: int = 8, dropout: float = 0.0, eps: float = 1e-7) -> None:
        super().__init__()

        self.eps = eps
        self.dropout = dropout
        self.num_heads = num_heads
        self.convs = ModuleList()
        self.norms = ModuleList()

        self.mlp = MLP(channels=[hidden_channels, out_channels*num_heads, out_channels])

        for i in range(num_heads):
            self.convs.append(GGAConv(hidden_channels, hidden_channels, hidden_channels))
            self.norms.append(BatchNorm1d(hidden_channels, affine=True))

    def forward(self, *args, **kwargs) -> Tensor:
        args = list(args)
        x: Tensor = args.pop(0)
        _x: List[Tensor] = []

        for i in range(self.num_heads):
            intter = self.convs[i](x, *args, **kwargs)
            intter = self.norms[i](intter)
            intter = F.relu(intter) + self.eps
            intter = F.dropout(intter, p=self.dropout, training=self.training)

            _x.append(intter)

        return self.mlp(sum(_x))