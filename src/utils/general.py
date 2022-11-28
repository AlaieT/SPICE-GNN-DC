__author__ = 'Alaie Titor'
__all__ = ['generate_folders', 'get_model_size']

import torch
import torch.utils.data
import numpy as np
import os


def get_model_size(model: torch.nn.Module):
    n_params = 0

    for _, param in model.named_parameters():
        n_params += np.prod(param.size())

    print(f'A total of {n_params:,} model parameters.\n')


def generate_folders(mode: str = 'train', **kwargs):
    if mode == 'train':
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        if not os.path.exists('./plots/metrik'):
            os.mkdir('./plots/metrik')
        if not os.path.exists('./dict'):
            os.mkdir('./dict')
            os.mkdir('./dict/dnn')
            os.mkdir('./dict/scaler')

    elif mode == 'generator':
        if not os.path.exists('./assets'):
            os.mkdir('./assets')
        if not os.path.exists(os.path.join('./assets', kwargs['gen_mode'])):
            os.mkdir(os.path.join('./assets', kwargs['gen_mode']))

    elif mode == 'analysis':
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        if not os.path.exists('./plots/model/'):
            os.mkdir('./plots/model/')

    elif mode == 'test':
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        if not os.path.exists('./plots/test'):
            os.mkdir('./plots/test')
