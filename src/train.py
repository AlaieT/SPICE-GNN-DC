import argparse
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from models import L2Error
import gc
from typing import List

from models import DeepIRDrop
from utils import CircuitDataset, plot_loss, generate_folders, top_k_acc, mape, get_model_size


def set_seed(
    seed: int = 42
):
    '''Set seed for every random generator that used in project'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    device: torch.device
):
    model.train()
    train_metrics = torch.empty((0, 4))

    for data in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.mask, data.batch)
        loss = criterion(out, data.target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        metrics = torch.Tensor([[loss.item(), top_k_acc(out, data.target, data.max_voltage, k=1), top_k_acc(out, data.target, data.max_voltage, k=2), mape(out, data.target)]])
        train_metrics = torch.cat([train_metrics, metrics], dim=0)

        del data, out, loss, metrics
        gc.collect()

    return train_metrics.mean(dim=0).tolist()


def valid_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    model.eval()
    valid_metrics = torch.empty((0, 3))

    with torch.no_grad():
        for data in tqdm(dataloader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            out = model(data.x, data.edge_index, data.mask, data.batch)

            metrics = torch.Tensor([[top_k_acc(out, data.target, data.max_voltage, k=1), top_k_acc(out, data.target, data.max_voltage, k=2), mape(out, data.target)]])
            valid_metrics = torch.cat([valid_metrics, metrics], dim=0)

            del data, out, metrics
            gc.collect()

    return valid_metrics.mean(dim=0).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-ft', '--file_train', default='./assets/train.csv', help='File path to train .csv file.')
    parser.add_argument('-fv', '--file_valid', nargs='+', default=['./assets/valid.csv'], help='File path to train .csv file.')
    parser.add_argument('-e', '--epochs', default=1500, type=int, help='Count of train epochs.')
    parser.add_argument('-bt', '--batch_size_train', default=128, type=int, help='Size of train batch.')
    parser.add_argument('-bv', '--batch_size_valid', default=128, type=int, help='Size of valid batch.')
    parser.add_argument('-r', '--resave', nargs='?', const=True, help='Resave train and batch data.')
    namespace = parser.parse_args()

    train_pack_path = namespace.file_train
    folds = namespace.file_valid
    batch_size_train = namespace.batch_size_train
    batch_size_valid = namespace.batch_size_valid
    epochs = namespace.epochs
    resave = namespace.resave

    assert os.path.exists(train_pack_path), 'Train dataset is not exists'
    for fold in folds:
        assert os.path.exists(fold), f'Path to {fold} does not exists!'
    assert batch_size_train > 0, 'Batch size not allowed to be 0'
    assert batch_size_valid > 0, 'Batch size not allowed to be 0'
    assert epochs > 0, 'Epochs not allowed to be 0'

    generate_folders(mode='train')
    set_seed(seed=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepIRDrop(in_channels=8, hidden_channels=48, out_channels=1, num_gcl=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-6)
    criterion = L2Error()

    metric = np.zeros((0, 4 + 3*len(folds)))
    best_loss = None

    get_model_size(model)

    dataset_train = CircuitDataset(train_pack_path, resave=resave, train=resave, device=device, scaler_path='./dict/scaler/min_max.joblib')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)

    folds_dataloders: List[DataLoader] = []
    for fold in folds:
        dataset_valid_out = CircuitDataset(fold, resave=resave, train=False, device=device, scaler_path='./dict/scaler/min_max.joblib')
        folds_dataloders.append(DataLoader(dataset=dataset_valid_out, batch_size=batch_size_valid, shuffle=False))

    print(f'\nStart traning on {epochs} epochs\n')

    for i in range(epochs):
        train_mean_acc = train_step(model, criterion, optimizer, scheduler, dataloader_train, device)

        folds_mean_acc = []
        for dataloader in folds_dataloders:
            folds_mean_acc += valid_step(model, dataloader, device)

        metric = np.append(metric, [train_mean_acc + folds_mean_acc], axis=0)

        if best_loss == None or best_loss >= metric[-1, 0]:
            best_loss = metric[-1, 0]
            torch.save(model.state_dict(), './dict/dnn/best.pt')
        else:
            torch.save(model.state_dict(), './dict/dnn/last.pt')

        print(f'\n---> {i+1}\033[95m LR:\033[0m {optimizer.param_groups[0]["lr"]:3e}')
        print(f'| Train \033[94mL2Error:\033[0m {metric[-1, 0]:.5}')
        print(f'| Train \033[94mAcc@1:\033[0m {metric[-1, 1]:.5}% \033[94mAcc@2\033[0m: {metric[-1, 2]:.5}% \033[94mMAPE\033[0m: {metric[-1, 3]:.5}%')

        for idx in range(len(folds)):
            print(f'| Fold{idx} \033[96mAcc@1:\033[0m {metric[-1, 4 + idx*3]:.5}% \033[96mAcc@2\033[0m: {metric[-1, 5 + idx*3]:.5}% \033[96mMAPE\033[0m: {metric[-1, 6 + idx*3]:.5}%')

        print('\033[94m--------------------------------------------------------------------------------\033[0m')

        if (i + 1) % 2 == 0:
            plot_loss(dataset=metric[:, :1].T, titles=['train'], save_path='./plots/metrik/train.png')
            plot_loss(dataset=metric[:, [1, ] + [4 + idx*3 for idx in range(len(folds))]].T,
                      titles=['train acc@1 %'] + [f'fold{idx} acc@1 %' for idx in range(len(folds))], save_path='./plots/metrik/acc1.png')
            plot_loss(dataset=metric[:, [2, ] + [5 + idx*3 for idx in range(len(folds))]].T,
                      titles=['train acc@2 %'] + [f'fold{idx} acc@2 %' for idx in range(len(folds))], save_path='./plots/metrik/acc2.png')
            plot_loss(dataset=metric[:, [3, ] + [6 + idx*3 for idx in range(len(folds))]].T,
                      titles=['train mape %'] + [f'fold{idx} mape %' for idx in range(len(folds))], save_path='./plots/metrik/mape.png')
