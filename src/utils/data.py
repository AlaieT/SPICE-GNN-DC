__author__ = 'Alaie Titor'
__all__ = ['CircuitDataset']

from typing import Any
from torch_geometric.typing import OptTensor

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

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
    def __init__(self, files_path: pd.DataFrame, train: bool = True, resave: bool = False, min_max_scaler_path: str = './dict/scaler/min_max.joblib', device: torch.device = torch.device('cpu')):
        super().__init__()
        self.data = []

        min_max_scaler = MinMaxScaler(feature_range=(0, 1)) if train else load(min_max_scaler_path)
        fit_x = torch.empty((0, 8)) if train else None

        for _, row in tqdm(files_path.iterrows(), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', total=len(files_path)):
            x, target, edge_index, mask, max_voltage = spice_to_graph(row['Source'], row['Target'], resave=resave)

            if train:
                fit_x = torch.cat([fit_x, x], dim=0)

            self.data.append(CircuitData(x=x, edge_index=edge_index, target=target, mask=mask, max_voltage=max_voltage).to(device))

        if train:
            min_max_scaler.fit(fit_x.cpu().numpy())

        for data in self.data:
            data.x = torch.tensor(min_max_scaler.transform(data.x.cpu().numpy()), dtype=torch.float, device=device)

        if train:
            dump(min_max_scaler, min_max_scaler_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        return self.data[idx]
