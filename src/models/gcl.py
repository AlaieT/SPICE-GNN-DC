__author__ = 'Alaie Titor'
__all__ = ['Extractor', 'Decoder', 'Encoder']


import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing, GraphNorm, MessageNorm
from .mlp import MLP


class GGAConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str = 'powermean',
            learn_param: bool = True,
            param: float = 1.0,
            eps: float = 1e-5):

        aggr_kwargs = {}
        if aggr == 'softmax':
            aggr_kwargs = dict(t=param, learn=learn_param)
        if aggr == 'powermean':
            aggr_kwargs = dict(p=param, learn=learn_param)

        super().__init__(aggr=aggr, aggr_kwargs=aggr_kwargs)

        channels = [in_channels, in_channels, out_channels]

        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.msg_norm = MessageNorm(False)
        self.mlp = MLP(channels=channels)

    def forward(self, x: Tensor, edge_index: Tensor, *args, **kwargs) -> Tensor:
        x = (x, x)
        out = self.propagate(x=x, edge_index=edge_index)
        out = self.msg_norm(x[0], out)
        out = out + x[1]

        return self.mlp(out, *args, **kwargs)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return x_j.relu() + self.eps


class DeepGGALayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, block_size: int = 2, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps
        self.block_size = block_size
        self.hidden_channels = hidden_channels
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.exp_lin = MLP(channels=[hidden_channels, hidden_channels * 2])

        for i in range(block_size):
            self.convs.append(GGAConv(hidden_channels, hidden_channels))
            # self.norms.append(GraphNorm(hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor, *args, **kwargs) -> Tensor:
        h = x

        for i in range(self.block_size):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            # x = self.norms[i](x, *args, **kwargs)

            if i < self.block_size - 1:
                x = F.relu(x) + self.eps

        x = h + x

        x = self.exp_lin(x)
        x = F.relu(x) + self.eps

        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps
        self.mlp = MLP(channels=[in_channels, out_channels])

    def forward(self, x: Tensor,  *args, **kwargs) -> Tensor:
        x = self.mlp(x, *args, **kwargs)

        return x


class Extractor(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 2) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.layers = ModuleList()

        for i in range(1, self.num_layers + 1):
            self.layers.append(DeepGGALayer(hidden_channels, block_size=4))
            hidden_channels = hidden_channels * 2

    def forward(self, x: Tensor, edge_index: Tensor, *args, **kwargs) -> Tensor:
        for i in range(self.num_layers):
            if x.requires_grad:
                x = checkpoint(self.layers[i], x, edge_index, *args, **kwargs)
            else:
                x = self.layers[i](x, edge_index, *args, **kwargs)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()

        self.mlp = MLP(channels=[hidden_channels, hidden_channels * 4, out_channels], norm='pair')

    def forward(self, x: Tensor, edge_index: Tensor, *args, **kwargs) -> Tensor:
        x = self.mlp(x, edge_index, *args, **kwargs)

        return x
