import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.py as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid

def visualise_dataset(train_loader, grayscale=True):
    dataiter = iter(train_loader)
    image = next(dataiter)

    num_samples = 25

    if grayscale:
        sample_images = [image[0][i,0] for i in range(num_samples)]
    else:
        sample_images = [image[0][i].permute(1, 2, 0) for i in range(num_samples)]

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        if grayscale:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im.numpy())
        ax.axis('off')

    plt.show()