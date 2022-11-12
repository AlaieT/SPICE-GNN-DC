import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models import L2Error

from models import DeepIRDrop
from utils import CircuitDataset, plot_loss, generate_folders


def set_seed(seed=42):
    '''Set seed for every random generator that used in project'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def top_k_acc(y_pred: torch.Tensor, y_true: torch.Tensor, max_voltage: torch.Tensor, k: int = 1):

    max_voltage = max_voltage.view(-1).detach().cpu().numpy()
    y_pred = y_pred.view(-1).detach().cpu().numpy()
    y_true = y_true.view(-1).detach().cpu().numpy()

    diff = np.around(np.abs(np.abs(y_pred - y_true)), k)
    loss = np.where(diff * max_voltage > 0.0, 1.0, 0.0)

    return np.mean(loss) * 100


def mape(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = y_pred.view(-1).detach().cpu().numpy()
    y_true = y_true.view(-1).detach().cpu().numpy()

    return mean_absolute_percentage_error(y_true, y_pred) * 100


def train_step(model, criterion, optimizer, scheduler, dataloader, device):
    model.train()
    train_acc = np.zeros((0, 4))

    for data in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        optimizer.zero_grad()

        target, max_voltage = data.target, data.max_voltage
        out = model(x=data.x, edge_index=data.edge_index, mask=data.mask)
        loss = criterion(out, target)

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)

        optimizer.step()
        scheduler.step()

        train_acc = np.append(train_acc, [[loss.item(), top_k_acc(out, target, max_voltage, k=1), top_k_acc(out, target, max_voltage, k=2), mape(out, target)]], axis=0)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.mean(train_acc, axis=0)


def valid_step(model, dataloader, device):
    model.eval()
    valid_acc = np.zeros((0, 3))

    with torch.no_grad():
        for data in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            target, max_voltage = data.target, data.max_voltage
            out = model(x=data.x, edge_index=data.edge_index, mask=data.mask)

            valid_acc = np.append(valid_acc,[[top_k_acc(out, target, max_voltage, k=1), top_k_acc(out, target, max_voltage, k=2), mape(out, target)]], axis=0)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.mean(valid_acc, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-f', '--file', default='./assets/train_ibm.csv', help='File path to train .csv file.')
    parser.add_argument('-e', '--epochs', default=1500, type=int, help='Count of train epochs.')
    parser.add_argument('-bt', '--batch_size_train', default=128, type=int, help='Size of train batch.')
    parser.add_argument('-bv', '--batch_size_valid', default=128, type=int, help='Size of valid batch.')
    parser.add_argument('-r', '--resave', nargs='?', const=True, help='Resave train and batch data.')
    namespace = parser.parse_args()

    train_pack_path = namespace.file
    batch_size_train = namespace.batch_size_train
    batch_size_valid = namespace.batch_size_valid
    epochs = namespace.epochs
    resave = namespace.resave

    assert os.path.exists(train_pack_path), 'Train dataset is not exists'
    assert batch_size_train > 0, 'Batch size not allowed to be 0'
    assert batch_size_valid > 0, 'Batch size not allowed to be 0'
    assert epochs > 0, 'Epochs not allowed to be 0'

    generate_folders(mode='train')
    set_seed(seed=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepIRDrop(in_channels=8, hidden_channels=64, out_channels=1, num_gcl=4, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-5)
    criterion = L2Error()
    metrick = np.zeros((0, 7))
    best_loss = None
    bad_steps = 0

    train_data, valid_data = train_test_split(pd.read_csv(train_pack_path), train_size=0.8, test_size=0.2, shuffle=True, random_state=42)

    dataset_train = CircuitDataset(train_data, resave=resave, train=resave, device=device)
    dataset_valid = CircuitDataset(valid_data, resave=resave, train=False, device=device)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size_valid, shuffle=False)

    for i in range(epochs):

        train_mean_acc = train_step(model, criterion, optimizer, scheduler, dataloader_train, device)

        valid_mean_acc = valid_step(model, dataloader_valid, device)

        metrick = np.append(metrick, [train_mean_acc.tolist() + valid_mean_acc.tolist()], axis=0)

        print(
            f'\n---> {i+1}\033[95m LR:\033[0m {optimizer.param_groups[0]["lr"]:3e}' +
            f'\n|\033[94m Train L2Error:\033[0m {metrick[-1, 0]:.5}' +
            f'\n|\033[94m Train Acc@1:\033[0m {metrick[-1, 1]:.5}% Acc@2: {metrick[-1, 2]:.5}% MAPE: {metrick[-1, 3]:.5}%' +
            f'\n|\033[96m Valid Acc@1:\033[0m {metrick[-1, 4]:.5}% Acc@2: {metrick[-1, 5]:.5}% MAPE: {metrick[-1, 6]:.5}%' +
            '\n\033[94m--------------------------------------------------------------------------------\033[0m')

        if (i + 1) % 5 == 0:
            plot_loss(dataset=metrick[:, :1].T, titles=['train'], save_path='./plots/metrik/train.png')
            plot_loss(dataset=metrick[:, [1, 4]].T, titles=['train acc@1 %', 'valid acc@1 %'], save_path='./plots/metrik/acc1.png')
            plot_loss(dataset=metrick[:, [2, 5]].T, titles=['train acc@2 %', 'valid acc@2 %'], save_path='./plots/metrik/acc2.png')
            plot_loss(dataset=metrick[:, [3, 6]].T, titles=['train mape %', 'valid mape %'], save_path='./plots/metrik/mape.png')

        if best_loss == None or best_loss >= metrick[-1, 0]:
            torch.save(model.state_dict(), './dict/dnn/best.pt')
        else:
            torch.save(model.state_dict(), './dict/dnn/last.pt')