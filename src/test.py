__author__ = 'Alaie Titor'

import os
import argparse
import numpy as np
from typing import List
from models import DeepIRDrop
from utils import CircuitDataset, top_k_acc, mape, generate_folders
from torch_geometric.loader import DataLoader
import torch
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing model performanse')
    parser.add_argument('-f', '--folds', nargs='?', type=str, default='', help='Path to folds. Can hold several paths.')
    parser.add_argument('-mp', '--model_path', nargs='?', type=str, default='./dict/dnn/best.pt', help="Path to the model dict.")
    parser.add_argument('-bs', '--batch_size', nargs=1, type=int, default='')
    parser.add_argument('-r', '--resave', nargs='?', type=bool, const=True, help='Resave tesnors for each fold.')
    namspace = parser.parse_args()

    folds = namspace.folds.split(' ')
    model_path = namspace.model_path
    batch_size = namspace.batch_size
    resave = namspace.resave

    '''Checking path existance'''
    for fold in folds:
        assert os.path.exists(folds), f"Fold path: {fold} is not exists!"
    assert os.path.exists(model_path), f"Model path: {model_path} is not exists!"
    assert batch_size > 0, "Batch size must be more then 0!"

    generate_folders(mode='test')

    '''List to hold metriks of all provided folds'''
    folds_metriks: List[np.ndarray] = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepIRDrop(in_channels=8, hidden_channels=64, out_channels=1, num_gcl=4).eval().to(device)
    model.load_state_dict(torch.load(model_path))

    '''List to hold all folds dataloaders'''
    dataloaders: List[DataLoader] = []

    for fold in folds:
        print(f'Creating dataloader for path: {fold}\n')

        dataset = CircuitDataset(files_path=pd.read_csv(fold), resave=resave, train=False, device=device)
        dataloaders.append(DataLoader(dataset=CircuitDataset, batch_size=batch_size, shuffle=False))

    '''Collecting test data'''
    for dataloader in dataloaders:
        fold_metrik = np.array((0, 3))

        for data in dataloader:
            out = model(x=data.x, edge_index=data.edge_index, mask=data.mask)
            fold_metrik = np.append(fold_metrik, [[top_k_acc(out, data.target, k=1), top_k_acc(out, data.target, k=2), mape(out, data.target)]], axis=0)

        folds_metriks.append(fold_metrik, axis=0)

    for idx, metrik in enumerate(folds_metriks):
        print(f"Fold_{idx}: Acc@1 - {metrik[0]} Acc@2 - {metrik[1]} MAPE - {metrik[2]}")
