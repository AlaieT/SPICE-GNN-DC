import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import gc
import pandas as pd
from typing import List, Any, Tuple

from models import DeepIRDrop, L2Error
from utils import (CircuitDataset, plot_surface_of_loss, generate_folders, mape, plot_any_surface)


def init_directions(model: torch.nn.Module, device: torch.device):
    noises = []

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

    return noises


def init_network(model: torch.nn.Module, all_noises: List, alpha: float, beta: float):
    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises

            new_value = param + alpha * delta + beta * nu
            param.copy_(new_value)

    return model


def generated_data_analysis(path: str):
    df = pd.read_csv(path)

    targets = df.iloc[:, 1]
    cells = df.iloc[:, 2]
    voltage = df.iloc[:, 3]

    print(f'\nData length: {df.shape[0]}')
    print(f'Cells - mean: {np.int0(np.mean(cells))} max: {np.max(cells)} min: {np.min(cells)}')
    print(f'Source Voltage - mean: {np.mean(voltage):.5} max: {np.max(voltage):.5} min: {np.min(voltage):.5}')

    diff = np.array([])
    min_voltage = None
    max_voltage = 0

    for target in targets:
        if not os.path.exists(target):
            raise ValueError(f"Can't open file: {target}")

        df_target = pd.read_csv(target)
        values = df_target.iloc[:, 1]

        if np.max(values) != 0:
            diff = np.append(diff, np.abs((np.max(values) - np.min(values[values > 0])) / np.max(values)))

            if np.max(values) > max_voltage:
                max_voltage = np.max(values)

            if np.min(values[values > 0]) < min_voltage or min_voltage is None:
                min_voltage = np.min(values[values > 0])

    print(f'Diff - mean: {np.mean(diff)*100:.5}% max: {np.max(diff)*100:.5}% min: {np.min(diff)*100:.5}%')
    print(f'Dropout Voltage - max: {max_voltage:.5} min: {min_voltage:.5}\n')


def loss_surface_analysis(model: torch.nn.Module, criterion: Any, data_path: str, model_path: str, resolution: int = 25, batch_size: int = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    criterion = L2Error()

    dataset = CircuitDataset(data_path, resave=False, train=False, device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(model_path))
    noises = init_directions(model, device)

    A, B = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution), indexing='ij')
    loss_surface = np.empty_like(A)

    for i in tqdm(range(resolution), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        for j in range(resolution):
            total_loss = 0
            n_batch = 0
            alpha = A[i, j]
            beta = B[i, j]

            model.load_state_dict(torch.load(model_path))
            net = init_network(model, noises, alpha, beta)

            for data in dataloader:
                with torch.no_grad():
                    out = net(x=data.x, edge_index=data.edge_index, mask=data.mask, batch=data.batch)
                    loss = criterion(out, data.target)
                    total_loss += loss.item()
                    n_batch += 1

            loss_surface[i, j] = total_loss / (n_batch * batch_size)

            del net, data, out
            gc.collect()

    plot_surface_of_loss('./plots/model/', 'deep_nn_loss', A, B, loss_surface)


def folds_analysis(folds: List[str], model: torch.nn.Module, criterion: Any, model_path: str, labels: Tuple[str, str, str] = ['Cells', 'Voltage', 'MAPE'], save_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    model.load_state_dict(torch.load(model_path))

    df = pd.read_csv(folds.pop(0))

    for fold in folds:
        df = pd.concat([df, pd.read_csv(fold)])

    data_x = df['Cells'].sort_values().unique()
    data_y = df['Voltage'].sort_values().unique()
    matrix = np.zeros((data_x.shape[0], data_y.shape[0]))
    df = df.groupby(['Cells', 'Voltage'])

    i = 0
    j = 0

    cell = data_x[0]
    voltage = data_y[0]

    with torch.no_grad():
        for _, group in tqdm(df, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            dataset = CircuitDataset(file=group, resave=False, train=False, show_progress=False, device=device)
            dataloader = DataLoader(dataset=dataset, batch_size=group.shape[0], shuffle=False)

            group_loss = []

            for data in dataloader:
                out = model(x=data.x, edge_index=data.edge_index, mask=data.mask, batch=data.batch)
                loss = criterion(out, data.target)

                group_loss.append(loss)

            i = i + 1 if cell < group.iloc[0]['Cells'] else i
            j = j + 1 if voltage < group.iloc[0]['Voltage'] else 0

            cell = group.iloc[0]['Cells']
            voltage = group.iloc[0]['Voltage']

            matrix[i, j] = sum(group_loss)/len(group_loss)

    plot_any_surface(data_x, data_y, matrix, labels=labels, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of generated data and model')
    parser.add_argument('-gd', '--generated_data', nargs='?', const=True, help='Analysis of generated data. Requres path to .csv file of generated data.')
    parser.add_argument('-ls', '--loss_surface', nargs='?', const=True, help='Analysis of dependencies of train loss function from model parametres. Requires path to the dataset .csv')
    parser.add_argument( '-fd', '--fold_dependings', nargs='?', const=True, help='Analysis of folds error depending on cells and voltage numbers. Requires path to the 4 datasets - all-in-range; voltage-out-of-range; cells-out-of-range; all-out-of-range.')
    parser.add_argument('-p', '--path', nargs='+', default=[], help='Target file or folder to analysis')
    parser.add_argument('-m', '--model', nargs='?', default='./dict/dnn/best.pt', help='Path to the model checkpoint.')
    namespace = parser.parse_args()

    generated_data = namespace.generated_data
    loss_surface = namespace.loss_surface
    fold_dependings = namespace.fold_dependings
    path = namespace.path
    model_path = namespace.model

    generate_folders(mode='analysis')

    if generated_data and path:
        if not os.path.exists(path[0]):
            raise ValueError('Path does not exists!')

        generated_data_analysis(path[0])

    elif loss_surface and path:
        if not os.path.exists(path[0]):
            raise ValueError('Path does not exists!')

        if not os.path.exists(model_path):
            raise ValueError('Model is not exists!')

        model = DeepIRDrop(in_channels=8, hidden_channels=48, out_channels=1, num_gcl=3)
        criterion = L2Error()

        loss_surface_analysis(model, criterion, path[0], model_path, resolution=25, batch_size=64)

    elif fold_dependings:
        for fold in path:
            if not os.path.exists(fold):
                raise ValueError(f'Path: {fold} does not exists!')

        if not os.path.exists(model_path):
            raise ValueError('Model is not exists!')

        model = DeepIRDrop(in_channels=8, hidden_channels=48, out_channels=1, num_gcl=3)
        criterion = mape

        folds_analysis(path, model, criterion, model_path)
