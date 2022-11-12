__author__ = 'Alaie Titor'
__all__ = ['plot_loss', 'plot_circuit_heatmap', 'plot_grad_flow', 'plot_surface_of_loss']

import argparse
import re
from cmath import inf
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os

def _fit_and_mask_heatmap_(heatmap, max_in_col) -> np.ndarray:
    for col in heatmap:
        if(len(col) < max_in_col):
            col += [0]*(max_in_col - len(col))

    heatmap = np.array(heatmap).T
    heatmap = np.ma.masked_where(heatmap == 0, heatmap)

    return heatmap


def _sort_by_coords(col: pd.Series):
    regex = re.compile('-?[\d.]+(?:e-?\d+)?')
    new_col = []

    for item in col:
        numbers = re.findall(regex, item)
        
        if item != '0':
            if float(numbers[1]) < 100000:
                numbers[1] = '0'*(6 - len(numbers[1])) + numbers[1]
            if float(numbers[2]) < 100000:
                numbers[2] = '0'*(6 - len(numbers[2])) + numbers[2]

            item =''.join(numbers)
        new_col.append(item)

    return pd.Series(new_col, name='Node')


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.grad != None):
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.cpu().abs().mean())
                max_grads.append(p.grad.cpu().abs().max())
        else:
            print(n)
            
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.05, lw=1, color="b")
    plt.bar(np.arange(len(ave_grads)), max_grads, alpha=0.05, lw=1, color="c")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1.0)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('./plots/grad.png', dpi = 200)


def plot_loss(dataset: np.ndarray, titles: list[str] = None, save_path: str = None):
    '''Plot graphs of provided data, created specialy for loss data...'''

    fig, axs = plt.subplots()

    for idx in range(dataset.shape[0]):
        axs.plot(np.linspace(start=0, stop=dataset.shape[1], num=dataset.shape[1]), dataset[idx])

    axs.legend(titles)
    fig.tight_layout()
    
    axs.grid(visible=True)

    if save_path is not None:
        plt.savefig(save_path, dpi = 200)
    else:
        plt.show()

    plt.close()


def plot_circuit_heatmap(filename: str, save_path: str = None):
    '''Plot heat map of curcuit solution'''

    regex = re.compile('-?[\d.]+(?:e-?\d+)?')
    nodes = pd.read_csv(filename)
    nodes.sort_values(by='Node', inplace=True, ascending=True, key= _sort_by_coords)
    nodes = np.expand_dims(nodes.iloc[:,:].to_numpy(), axis=1)

    layer = None
    pos_x = None

    layers = []
    heatmap = []
    col = []

    max_in_col = 0
    max_v = 0
    min_v = +inf

    for idx, node in enumerate(nodes):
        if node[0, 0] != '0' and '_X_' not in node[0,0]:
            node_numbers = re.findall(regex, node[0,0])

            if layer is None:
                layer = int(node_numbers[0])

            elif layer != int(node_numbers[0]):
                if len(col) > max_in_col:
                    max_in_col = len(col)

                heatmap.append(copy(col))
                layers.append(_fit_and_mask_heatmap_(heatmap, max_in_col))
                layer = int(node_numbers[0])
                pos_x = int(node_numbers[1])
                heatmap = []
                col = []

            if pos_x is None:
                pos_x = int(node_numbers[1])

            elif pos_x != int(node_numbers[1]):
                if len(col) > max_in_col:
                    max_in_col = len(col)

                pos_x = int(node_numbers[1])
                heatmap.append(copy(col))
                col = []

            col.append(node[0, 1])

            if max_v < node[0, 1]:
                max_v = node[0, 1]

            if min_v > node[0, 1]:
                min_v = node[0, 1]

        if idx == nodes.shape[0] - 1:
            if len(col) > max_in_col:
                    max_in_col = len(col)

            heatmap.append(copy(col))
            layers.append(_fit_and_mask_heatmap_(heatmap, max_in_col))

    print(f'Is min value negative?: {True if min_v < 0 else False}')
    print(f'Max-min diff: {(1 - min_v / max_v) * 100:.5}%\n')
    
    fig, axes = plt.subplots(1, len(layers))

    fig.set_figwidth(20)
    fig.set_figheight(10)

    if len(layers) > 1:
        for idx, layer in enumerate(layers):
            sns.heatmap(layer, cmap="rocket", vmin=min_v, vmax=max_v, ax=axes[idx], cbar= not (idx + 1) % len(layers))
            axes[idx].set_title(f"Layer {idx+1}")
    else:
        sns.heatmap(layers[0], cmap="rocket", vmin=min_v, vmax=max_v,)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi = 400)
    else:
        plt.show()

    plt.close()


def plot_surface_of_loss(save_path, model, xx, yy, zz):
    zz = np.log(zz)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, zz)
    plt.savefig(os.path.join(save_path, f'{model}_log_contour.png'), dpi=100)
    plt.close()

    ## 3D plot
    fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
    ax.set_axis_off()
    ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.savefig(os.path.join(save_path, f'{model}_log_surface.png'), dpi=100, format='png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    # ax.set_axis_off()

    def init():
        ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    def animate(i):
        ax.view_init(elev=(15 * (i // 15) + i % 15) + 0., azim=i)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)

    anim.save(os.path.join(save_path,f'{model}_log_surface.gif'), fps=15,  writer='imagemagick')