__author__ = 'Alaie Titor'
__all__ = ['plot_loss', 'plot_grad_flow', 'plot_surface_of_loss', 'plot_test']

import argparse
import re
from cmath import inf
from copy import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os
import torch


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
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=1.0)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('./plots/grad.png', dpi=200)


def plot_loss(dataset: np.ndarray, titles: list[str] = None, save_path: str = None):
    '''Plot graphs of provided data, created specialy for loss data...'''

    fig, axs = plt.subplots()

    for idx in range(dataset.shape[0]):
        axs.plot(np.linspace(start=0, stop=dataset.shape[1], num=dataset.shape[1]), dataset[idx])

    axs.legend(titles)
    axs.grid(visible=True)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()


def plot_test(data_x: np.ndarray, data_y: np.ndarray, data_z: np.ndarray, save_path: str = None):
    sns.heatmap(data=data_z, yticklabels=data_y, xticklabels=data_x, cmap='rocket')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_surface_of_loss(save_path: str, model: torch.nn.Module, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray):
    zz = np.log(zz)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, zz)
    plt.savefig(os.path.join(save_path, f'{model}_log_contour.png'), dpi=100)
    plt.close()

    # 3D plot
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
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

    anim.save(os.path.join(save_path, f'{model}_log_surface.gif'), fps=15,  writer='imagemagick')
