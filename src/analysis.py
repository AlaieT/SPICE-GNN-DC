import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

from models import DeepIRDrop, L2Error
from utils import (CircuitDataset, init_directions, init_network, plot_surface_of_loss, generated_data_info, plot_circuit_heatmap, generate_folders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of generated data and model')
    parser.add_argument('-gd', '--generated_data', nargs='?', const=True, help='Analysis of generated data. Requres path to .csv file of generated data.')
    parser.add_argument('-ch', '--circuit_heatmap', nargs='?', const=True, help='Analysis of circuit voltage dropout using heatmap. Requires path to train dataset .csv file.')
    parser.add_argument('-ls', '--loss_surface', nargs='?', const=True, help='Analysis of dependencies of train loss function from model parametres. Requires path to the dataset .csv')
    parser.add_argument('-p', '--path', nargs='?', default='', help='Target file or folder to analysis')
    namespace = parser.parse_args()

    generated_data = namespace.generated_data
    circuit_heatmap = namespace.circuit_heatmap
    loss_surface = namespace.loss_surface
    path = namespace.path

    assert os.path.exists(path), 'Path is not exists!'

    generate_folders(mode='analysis')

    if generated_data and path:
        generated_data_info(file_name=path)

    elif circuit_heatmap and path:
        plot_circuit_heatmap(filename=path, save_path=f'./plots/circuit/{path.split("/")[-1][:-4]}')

    elif loss_surface and path:
        if os.path.exists('./dict/dnn/best.pt'):
            RESOLUTION = 25
            BATCH_SIZE = 256

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DeepIRDrop(in_channels=8, hidden_channels=64, out_channels=1, num_gcl=4).eval().to(device)
            criterion = L2Error()

            _, data = train_test_split(pd.read_csv(path), train_size=0.8, test_size=0.2, shuffle=True, random_state=42)
            dataset = CircuitDataset(data, resave=False, train=False)
            dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            model.load_state_dict(torch.load('./dict/dnn/best.pt'))
            noises = init_directions(model, device)

            A, B = np.meshgrid(np.linspace(-1, 1, RESOLUTION), np.linspace(-1, 1, RESOLUTION), indexing='ij')
            loss_surface = np.empty_like(A)

            for i in tqdm(range(RESOLUTION), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                for j in range(RESOLUTION):
                    total_loss = 0
                    n_batch = 0
                    alpha = A[i, j]
                    beta = B[i, j]

                    model.load_state_dict(torch.load('./dict/dnn/best.pt'))
                    net = init_network(model, noises, alpha, beta)

                    for data in dataloader:
                        x, edge_index, target, mask, max_voltage = data.x.to(device), data.edge_index.to(device), data.target.to(device), data.mask.to(device), data.max_voltage

                        with torch.no_grad():
                            out = net(x=x, edge_index=edge_index, mask=mask)
                            loss = criterion(out, target)
                            total_loss += loss.item()
                            n_batch += 1

                    loss_surface[i, j] = total_loss / (n_batch * BATCH_SIZE)

                    del net, data, x, edge_index, target, mask, out

                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

            plot_surface_of_loss('./plots/model/', 'deep_nn_loss', A, B, loss_surface)

        else:
            print('Model dict is not exists!')
