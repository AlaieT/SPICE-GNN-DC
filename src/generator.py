__author__ = 'Alaie Titor'

import math
import os
from random import uniform
from typing import List, Optional, Tuple

import pandas as pd

import argparse

from utils import generate_folders


class GridCell():
    def __init__(
        self,
        position: List[int],
        layer: int,
        res_value: float,
        via_value: float = 0.0,
        current_value: float = 0.0,
        volt_value: float = None,
        edges: Optional[List[int]] = [1, 1, 1, 1],
        current: Optional[List[int]] = [1, 1, 1, 1],
        next_layer: Optional[int] = None,
        vias: Optional[List[int]] = [1, 1, 1, 1],
        via_dropout: float = 0.0,
    ) -> None:

        via_drop_cf = uniform(0.0, 1.0)
        cell_res = [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0]]
        self.__cell_disc: List[str] = []

        for idx, (res, edge) in enumerate(zip(cell_res, edges)):
            if edge:
                if (res[0] == 1 or res[2] == 1) and volt_value is not None:
                    plus_l = f"n{layer}_{position[0] + res[0]}_{position[1] + res[1]}"
                    minus_l = f"n{layer}_{position[0] + res[2]}_{position[1] + res[3] + 0.5}"
                    name_l = f"Rl{layer}_{position[0] + res[0]}_{position[1] + res[1]}_{position[0] + res[2]}_{position[1] + res[3]}"

                    minus_r = f"n{layer}_{position[0] + res[2]}_{position[1] + res[3]}"
                    name_r = f"Rr{layer}_{position[0] + res[0]}_{position[1] + res[1]}_{position[0] + res[2]}_{position[1] + res[3]}"

                    vr_name = f"{layer}_{position[0] + res[0]}_{position[1] + res[1]}_{position[0] + res[2]}_{position[1] + res[3]}"

                    self.__cell_disc.append(f"{name_l} {plus_l} {minus_l} {res_value}\n")
                    self.__cell_disc.append(f"{name_r} {minus_l} {minus_r} {res_value}\n")
                    self.__cell_disc.append(f"rv{vr_name} {minus_l} _X_{minus_l} 0.25\n")
                    self.__cell_disc.append(f"vv{vr_name} _X_{minus_l} 0 {volt_value}\n")

                    volt_value = None
                else:
                    plus = f"n{layer}_{position[0] + res[0]}_{position[1] + res[1]}"
                    minus = f"n{layer}_{position[0] + res[2]}_{position[1] + res[3]}"
                    name = f"R{layer}_{position[0] + res[0]}_{position[1] + res[1]}_{position[0] + res[2]}_{position[1] + res[3]}"

                    self.__cell_disc.append(f"{name} {plus} {minus} {res_value}\n")

        if next_layer is not None and next_layer > layer:
            for res, via in zip(cell_res, vias):
                if via and via_drop_cf >= via_dropout:
                    plus = f"n{layer}_{position[0] + res[0]}_{position[1] + res[1]}"
                    minus = f"n{next_layer}_{position[0] + res[0]}_{position[1] + res[1]}"
                    name = f"V{layer}_{next_layer}_{position[0] + res[0]}_{position[1] + res[1]}"

                    self.__cell_disc.append(f"{name} {plus} {minus} {via_value}\n")

        if current is not None:
            for res, via in zip(cell_res, vias):
                if via:
                    plus = f"n{layer}_{position[0] + res[0]}_{position[1] + res[1]}"
                    minus = f"0"
                    name = f"iB_{layer}_{position[0] + res[0]}_{position[1] + res[1]}_v"

                    self.__cell_disc.append(f"{name} {plus} {minus} {current_value}\n")

    def get_disc(self) -> List[str]:
        return self.__cell_disc


def generate_data(
    mode: str = 'train',
    data_path: str = './assets/train/',
    volt_range: Tuple[int, int, int] = (500, 1000, 20),
    cells_range: Tuple[int, int, int] = (2, 6, 1),
    num_layers: int = 1,
    res_range: Tuple[int, int, int] = (10, 200, 10),
    via_drop=0.0
):

    edges = [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]]
    current = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]]
    vias = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 0]]

    files: List[str] = []
    count = 0

    for volt in range(volt_range[0], volt_range[1], volt_range[2]):
        for l in range(1, num_layers + 1):
            for x in range(cells_range[0], cells_range[1] + 1, cells_range[2]):
                for y in range(cells_range[0], cells_range[1] + 1, cells_range[2]):
                    for res_value in range(res_range[0], res_range[1], res_range[2]):
                        cells: List[GridCell] = []
                        size = [x, y, l]
                        next_layer = None
                        res_value = res_value/1000 * (1 - 0.2 * l / num_layers)
                        current_value = (volt/200)/((4*res_value + 3*(size[0]) * res_value + 3*(size[1]) * res_value + (size[0])*(size[1]) * 2 * res_value) * 2)
                        volt_value = volt/100

                        for z in range(size[-1]):
                            if z < size[-1] - 1:
                                next_layer = z + 1
                            else:
                                next_layer = None

                            for i in range(size[0]):
                                for j in range(size[1]):
                                    cell_volt = None
                                    cell_res = res_value * (1 + uniform(-0.15, 0.15))
                                    cell_current = current_value

                                    if i == 0 and j == 0:
                                        cells.append(
                                            GridCell(
                                                position=[i, j],
                                                layer=z,
                                                next_layer=next_layer,
                                                res_value=cell_res,
                                                current_value=cell_current,
                                                volt_value=cell_volt,
                                                edges=edges[0],
                                                current=current[0] if z == 0 else None,
                                                vias=vias[0],
                                                via_dropout=via_drop,
                                            ))
                                    elif i == 0:
                                        cells.append(
                                            GridCell(
                                                position=[i, j],
                                                layer=z,
                                                next_layer=next_layer,
                                                res_value=cell_res,
                                                current_value=cell_current,
                                                volt_value=cell_volt,
                                                edges=edges[1],
                                                current=current[1] if z == 0 else None,
                                                vias=vias[1],
                                                via_dropout=via_drop,
                                            ))
                                    elif j == 0:
                                        cells.append(
                                            GridCell(
                                                position=[i, j],
                                                layer=z,
                                                next_layer=next_layer,
                                                res_value=cell_res,
                                                current_value=cell_current,
                                                volt_value=cell_volt,
                                                edges=edges[2],
                                                current=current[2] if z == 0 else None,
                                                vias=vias[2],
                                                via_dropout=via_drop,
                                            ))
                                    else:
                                        cells.append(
                                            GridCell(
                                                position=[i, j],
                                                layer=z,
                                                next_layer=next_layer,
                                                res_value=cell_res,
                                                current_value=cell_current,
                                                volt_value=cell_volt,
                                                edges=edges[3],
                                                current=current[3] if z == 0 else None,
                                                vias=vias[3],
                                                via_dropout=via_drop,
                                            ))

                        if not os.path.exists(os.path.join(data_path, f'ibmpg{count}')):
                            os.mkdir(os.path.join(data_path, f'ibmpg{count}'))

                        with open(os.path.join(data_path, f'ibmpg{count}/ibmpg{count}.spice'), 'w') as file:
                            file.write(f'* {mode}_ibm_{count}\n')

                            file.write(f"rv0 n{size[-1]- 1}_0_0 _X_n{size[-1] - 1}_0_0 0.25\n")
                            file.write(f"vv0 _X_n{size[-1] - 1}_0_0 0 {volt_value}\n")

                            for idx, cell in enumerate(cells):
                                file.write(f'* Cell: {idx}\n')
                                file.writelines(cell.get_disc())

                            file.close()

                            files.append([os.path.join(data_path, f'ibmpg{count}/ibmpg{count}.spice'),
                                          os.path.join(data_path, f'ibmpg{count}/ibmpg{count}.csv'), size[0]*size[1], volt_value])

                            if os.path.exists(os.path.join(data_path, f'ibmpg{count}/ibmpg{count}.csv')):
                                os.remove(os.path.join(data_path, f'ibmpg{count}/ibmpg{count}.csv'))

                            count += 1

    df = pd.DataFrame(data=files, columns=['Source', 'Target', 'Cells', 'Voltage'])
    df.to_csv(os.path.join('./assets', f'{mode}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for circuit generation')
    parser.add_argument('-m', '--mode', default='train', help='Mod generation of dataset - train or test. Default = train',)
    parser.add_argument('-ct', '--circuit_type', default='grid', help='Type of circuit to be generated(grid or parallel). Default = grid')
    parser.add_argument('-vr', '--volt_range', nargs='+', type=int, default=[500, 1020, 20], help='Voltage range of curcuits. Default = (500, 1020, 20)',)
    parser.add_argument('-cr', '--cells_range', nargs='+', type=int, default=[1, 11, 1], help='Cells range of curcuits. Default = (1, 11, 1)',)
    parser.add_argument('-rr', '--res_range', nargs='+', type=int, default=[10, 410, 20], help='Resistance range of curcuits. Default = (10, 410, 20)',)
    parser.add_argument('-nl', '--num_layers', default=4, type=int, help='Number of layers of curcuits. Default = 4',)
    namespace = parser.parse_args()

    mode = namespace.mode
    circuit_type = namespace.circuit_type
    volt_range = tuple(namespace.volt_range)
    cells_range = tuple(namespace.cells_range)
    res_range = tuple(namespace.res_range)
    num_layers = namespace.num_layers

    assert isinstance(mode, str) == True, 'mode can be only string format!'
    assert len(volt_range) == 3, 'volt_range must have 3 parametres(min, max, step)!'
    assert len(cells_range) == 3, 'cells_range must have 3 parametres(min, max, step)!'
    assert len(res_range) == 3, 'res_range must have 3 parametres(min, max, step)!'
    assert isinstance(num_layers, int) == True, 'num_layers can be only int format!'

    generate_folders(mode='generator', gen_mode=mode)
    generate_data(mode=mode, data_path=os.path.join('./assets', mode), volt_range=volt_range, cells_range=cells_range, num_layers=num_layers, res_range=res_range, via_drop=0.0)
