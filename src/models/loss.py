__author__ = 'Alaie Titor'
__all__ = ['MAPE', 'RSME', 'L2Error']

import torch
from torch import Tensor

class MAPE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''MAPE - Mean Avarge Precentage Error'''
        input, target = input.view(-1), target.view(-1)
        loss = (input - target).abs() / target
        return loss.mean()


class RSME(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''MAPE - Root Mean Squared Error'''
        input, target = input.view(-1), target.view(-1)
        loss = (input - target).abs().pow(2).mean()

        return loss.sqrt()

class L2Error(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''Navier–Stokes equation'''
        input, target = input.view(-1), target.view(-1)
        loss = (input - target).pow(2).sum().sqrt() / target.pow(2).sum().sqrt()

        return loss
        