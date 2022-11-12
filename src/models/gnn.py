__author__ = 'Alaie Titor'
__all__ = ['DeepIRDrop']

import torch
from torch import Tensor

from .gcl import Extractor, GGAConv, Decoder


class DeepIRDrop(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_gcl: int = 2, dropout: float = 0.1) -> None:
        super().__init__()

        self.dropout = dropout
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.encoder = GGAConv(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels)
        self.extractor = Extractor(hidden_channels=hidden_channels, num_layers=num_gcl, dropout=dropout)
        self.decoder = Decoder(hidden_channels, out_channels, num_heads=8, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor, mask: Tensor) -> Tensor:
        x = self.encoder(x, edge_index)
        x = self.extractor(x, edge_index)
        x = self.decoder(x, edge_index)

        x = x.index_select(0, mask)

        return x
