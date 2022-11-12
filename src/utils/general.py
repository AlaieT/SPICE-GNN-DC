__author__ = 'Alaie Titor'
__all__ = ['generated_data_info', 'init_directions', 'init_network', 'generate_folders']

from cmath import inf
import torch
import numpy as np
import pandas as pd
import os

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
        if not os.path.exists('./plots/circuit/'):
            os.mkdir('./plots/circuit/')
        if not os.path.exists('./plots/model/'):
                os.mkdir('./plots/model/')


def generated_data_info(file_name:str):
    df = pd.read_csv(file_name)
    df = df.drop(columns='Unnamed: 0', axis=1)

    targets = df.iloc[:,1]
    cells = df.iloc[:,2]
    voltage = df.iloc[:,3]

    print(f'\nData length: {df.shape[0]}')
    print(f'Cells - mean: {np.int0(np.mean(cells))} max: {np.max(cells)} min: {np.min(cells)}')
    print(f'Source Voltage - mean: {np.mean(voltage):.5} max: {np.max(voltage):.5} min: {np.min(voltage):.5}')

    diff = np.array([])
    is_negative = np.array([])
    min_voltage = +inf
    max_voltage = 0

    for target in targets:
        if os.path.exists(target):
            dt = pd.read_csv(target)
            values = dt.iloc[:,1]

            if values[values < 0].shape[0] == 0:
                if np.max(values) != 0:
                    diff = np.append(diff, np.abs((np.max(values) - np.min(values[values > 0])) / np.max(values)))

                    if np.max(values) > max_voltage:
                        max_voltage = np.max(values)

                    if np.min(values[values > 0]) < min_voltage:
                        min_voltage = np.min(values[values > 0])
            else:
                is_negative = np.append(is_negative, target)
                df = df.drop(df[df.Target == target].index)

    print(f'Diff - mean: {np.mean(diff)*100:.5}% max: {np.max(diff)*100:.5}% min: {np.min(diff)*100:.5}%')
    print(f'Dropout Voltage - max: {max_voltage:.5} min: {min_voltage:.5}\n')
    
    for neg in is_negative:
        print(f"Filename: {neg}")


def init_directions(model, device):
    noises = []

    n_params = 0
    for _, param in model.named_parameters():
        delta = torch.normal(.0, 1., size=param.size()).to(device)
        nu = torch.normal(.0, 1., size=param.size()).to(device)

        param_norm = torch.norm(param)
        delta_norm = torch.norm(delta)
        nu_norm = torch.norm(nu)

        delta /= delta_norm
        delta *= param_norm

        nu /= nu_norm
        nu *= param_norm

        noises.append((delta, nu))

        n_params += np.prod(param.size())

    print(f'A total of {n_params:,} parameters.')

    return noises


def init_network(model, all_noises, alpha, beta):
    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises
            # the scaled noises added to the current filter
            new_value = param + alpha * delta + beta * nu
            param.copy_(new_value)
    return model