import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class VAE(nn.Module):
    """
    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim=400, latent_dim=200, device=torch.device("cpu")):
        super(VAE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, latent_dim),
        nn.LeakyReLU(0.2),
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

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