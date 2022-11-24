__author__ = 'Alaie Titor'
__all__ = ['DeepIRDrop']

import torch
from torch import Tensor

from .gcl import Extractor, Encoder, Decoder


class DeepIRDrop(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_gcl: int = 2) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.encoder = Encoder(in_channels=in_channels, out_channels=hidden_channels)
        self.extractor = Extractor(hidden_channels=hidden_channels, num_layers=num_gcl)
        self.decoder = Decoder(hidden_channels=hidden_channels * pow(2, num_gcl), out_channels=out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, mask: Tensor, batch: Tensor) -> Tensor:
        x = self.encoder(x, edge_index, batch)
        x = self.extractor(x, edge_index, batch)
        x = self.decoder(x, batch)

        x = x.index_select(0, mask)

        return x
