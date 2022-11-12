__author__ = 'Alaie Titor'
__all__ = ['spice_to_graph']

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

NODE_TYPES = {
    'n': 0,
    '_': 0,
    '0': 1,
    "R": 2,
    "r": 2,
    "v": 3,
    'i': 4
}


def _is_float(string: str) -> bool:
    r'''
    Check if string is correct float value

    :param string - string value that will be checked
    '''

    try:
        float(string)
        return True
    except:
        return False


def _exists(_dict: Dict[str, any], _key: str) -> bool:
    r'''
    Check if key exists in dict

    :param _dict - dictionary where the search will be performed
    :param _key - key that expcted to be found
    '''

    if(_dict.get(_key)):
        return True
    else:
        return False


class Node():
    def __init__(self, name, number, type, value) -> None:
        self.__name: str = name
        self.__number: int = number
        self.__type: int = type
        self.__value: float = value
        self.__connections: list[int] = []

    def get_name(self) -> str:
        '''Return name of the node'''
        return self.__name

    def get_number(self) -> int:
        '''Return number of the node'''
        return self.__number

    def get_type(self) -> int:
        '''Return type of the node'''
        return self.__type

    def get_value(self) -> float:
        '''Return value of the node'''
        return self.__value

    def get_connections(self) -> list[int]:
        '''Return all numbers of connected nodes'''
        return self.__connections

    def add_connection(self, connection: int):
        '''Add new node connection to the node'''
        if connection not in self.__connections:
            self.__connections.append(connection)

    def set_value(self, value):
        '''Set value of node'''
        self.__value = value


def spice_to_graph(source_filename: str, target_filename: Optional[str] = None, resave: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Read spice format file and return tensor view of circuit'''

    regex = re.compile('-?[\d.]+(?:[e]?[+-]?\d+)?')
    node_number = 0

    nodes: Dict[str, Node] = {}
    max_voltage = 0

    if not os.path.exists('/'.join(source_filename.split('/')[:-1] + ['tensor'])) or resave:
        with open(source_filename) as file:
            lines = file.readlines()

            for line in lines:
                if '*' not in line and '#' not in line:
                    splited_line = line.split(' ')
                    node_value = re.findall(regex, line)[-1]

                    if _is_float(node_value):

                        if not _exists(nodes, splited_line[1]):
                            nodes[splited_line[1]] = Node(splited_line[1], node_number, NODE_TYPES[splited_line[1][0]], 0.0)
                            node_number += 1

                        if not _exists(nodes, splited_line[2]):
                            nodes[splited_line[2]] = Node(splited_line[2], node_number, NODE_TYPES[splited_line[2][0]], 0.0)
                            node_number += 1

                        if line[0] != 'V':
                            if not _exists(nodes, splited_line[0]):
                                nodes[splited_line[0]] = Node(splited_line[0], node_number, NODE_TYPES[line[0]], float(node_value))
                                node_number += 1

                                if nodes[splited_line[0]].get_type() == NODE_TYPES['v'] and max_voltage < nodes[splited_line[0]].get_value():
                                    max_voltage = nodes[splited_line[0]].get_value()

                            nodes[splited_line[0]].add_connection(nodes[splited_line[1]].get_number())
                            nodes[splited_line[0]].add_connection(nodes[splited_line[2]].get_number())

                            nodes[splited_line[1]].add_connection(nodes[splited_line[0]].get_number())
                            nodes[splited_line[2]].add_connection(nodes[splited_line[0]].get_number())
                        else:
                            nodes[splited_line[1]].add_connection(nodes[splited_line[2]].get_number())
                            nodes[splited_line[2]].add_connection(nodes[splited_line[1]].get_number())

            file.close()

            x = np.zeros((len(nodes), 8))
            edge_index = np.zeros((0, 2))
            mask = np.array([])

            if target_filename is not None:
                data_target = pd.read_csv(target_filename)
                target = np.zeros((0, 1))

                for idx, node in enumerate(nodes.values()):

                    x[idx, node.get_type()] = 1.0

                    if node.get_type() == NODE_TYPES['R']:
                        x[idx, 5] = node.get_value()
                    elif node.get_type() == NODE_TYPES['v']:
                        x[idx, 6] = node.get_value()
                    elif node.get_type() == NODE_TYPES['i']:
                        x[idx, 7] = node.get_value()

                    edge_index = np.append(edge_index, [[node.get_number(), connection] for connection in node.get_connections()], axis=0)

                    if node.get_type() == NODE_TYPES['n']:
                        target = np.append(target, [[data_target.loc[data_target['Node'] == node.get_name()].to_numpy()[0][1]]], axis=0)
                        mask = np.append(mask, node.get_number())

                x = torch.tensor(x, dtype=torch.float)
                target = torch.tensor(target, dtype=torch.float)
                edge_index = torch.tensor(edge_index.T, dtype=torch.long)
                mask = torch.tensor(mask, dtype=torch.long)

                target = target / max_voltage

                max_voltage = torch.tensor(max_voltage)
                max_voltage = max_voltage.repeat(target.shape)

                if not os.path.exists('/'.join(source_filename.split('/')[:-1] + ['tensor'])):
                    os.mkdir('/'.join(source_filename.split('/')[:-1] + ['tensor']))

                torch.save(x, '/'.join(source_filename.split('/')[:-1] + ['tensor', 'x.plt']))
                torch.save(target, '/'.join(source_filename.split('/')[:-1] + ['tensor', 'target.plt']))
                torch.save(max_voltage, '/'.join(source_filename.split('/')[:-1] + ['tensor', 'max_voltage.plt']))
                torch.save(edge_index, '/'.join(source_filename.split('/')[:-1] + ['tensor', 'edge_index.plt']))
                torch.save(mask, '/'.join(source_filename.split('/')[:-1] + ['tensor', 'mask.plt']))

            return x, target, edge_index, mask, max_voltage

    else:
        x = torch.load('/'.join(source_filename.split('/')[:-1] + ['tensor', 'x.plt']))
        target = torch.load('/'.join(source_filename.split('/')[:-1] + ['tensor', 'target.plt']))
        max_voltage = torch.load('/'.join(source_filename.split('/')[:-1] + ['tensor', 'max_voltage.plt']))
        edge_index = torch.load('/'.join(source_filename.split('/')[:-1] + ['tensor', 'edge_index.plt']))
        mask = torch.load('/'.join(source_filename.split('/')[:-1] + ['tensor', 'mask.plt']))

        return x, target, edge_index, mask, max_voltage
