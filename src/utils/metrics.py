__author__ = 'Alaie Titor'
__all__= ['top_k_acc', 'mape']

from torch import Tensor
import torch


def top_k_acc(y_pred: Tensor, y_true: Tensor, max_voltage: Tensor, k: int = 1):

    max_voltage = max_voltage.detach()
    y_pred = y_pred.detach()
    y_true = y_true.detach()

    diff = torch.round(torch.abs(torch.abs(y_pred - y_true)), decimals=k)
    loss = torch.where(diff * max_voltage > 0.0, 1.0, 0.0)

    return loss.mean() * 100


def mape(y_pred: Tensor, y_true: Tensor):
    y_pred = y_pred.detach()
    y_true = y_true.detach()

    loss = (y_pred - y_true).abs() / y_true

    return loss.mean() * 100