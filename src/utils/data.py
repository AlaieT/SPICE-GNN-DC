__author__ = 'Alaie Titor'
__all__ = ['CircuitDataset']

from typing import Any, Union
from torch_geometric.typing import OptTensor

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import os

from .reader import spice_to_graph


class CircuitData(Data):
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face' or key == 'mask':
            return self.num_nodes
        else:
            return 0


class CircuitDataset(Dataset):
    def __init__(
        self,
        file: Union[str, pd.DataFrame],
        train: bool = True,
        resave: bool = False,
        show_progress: bool = True,
        scaler_path: str = './dict/scaler/min_max.joblib',
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.dataset = []

        if file is str and not os.path.exists(file):
            raise ValueError(f"Can't read from: {file}")

        if not os.path.exists(scaler_path):
            raise ValueError(f"Can't read from: {scaler_path}")

        data = pd.read_csv(file) if file is str else file
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5)) if train else load(scaler_path)
        fit_x = torch.empty((0, 8)) if train else None

        for _, row in tqdm(data.iterrows(), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(data), disable=not show_progress):
            x, target, edge_index, mask, max_voltage = spice_to_graph(row['Source'], row['Target'], resave=resave)
            self.dataset.append(CircuitData(x=x, edge_index=edge_index, target=target, mask=mask, max_voltage=max_voltage).to(device, non_blocking=True))

            if train:
                fit_x = torch.cat([fit_x, x], dim=0)

        if train:
            scaler.fit(fit_x.cpu().numpy())

        for data in self.dataset:
            data.x = torch.tensor(scaler.transform(data.x.cpu().numpy()), dtype=torch.float, device=device)

        if train:
            dump(scaler, scaler_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Data:
        return self.dataset[idx]
