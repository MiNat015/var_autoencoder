# Image Generation using an Autoencoder
PyTorch implementation of the Variational Autoencoder.

## Author
Arjun Srikanth (Final Year Student of Computer Science)

## Project Overview
![alt text](https://github.com/[MiNat015]/[var_autoencoder]/imgs/model.png?raw=true)
This project demonstrates the use of a Variational Autoencoder (VAE) to generate images. VAEs are powerful generative models that learn to encode data into a latent space, from which new data can be generated. The implementation is done using PyTorch, and the project includes training, data augmentation, and evaluation of the VAE on a standard image dataset.

## Variational Autoencoder
A Variational Autoencoder (VAE) is a generative model that learns the underlying distribution of the data. Unlike traditional autoencoders, which compress data into a fixed latent space, VAEs impose a probabilistic structure on the latent space. This allows for the generation of new data points by sampling from the learned distribution. The VAE consists of two main components: the encoder, which maps input data to the latent space, and the decoder, which reconstructs data from the latent space.

## Repository Overview
- `train.py`: The main script to train the VAE model.
- `modules.py`: Contains the implementation of Variational Autoencoder class along with a few utility functions for training.
- `slurm.sh`: Slurm script to train model on cluster.
- `README.md`: Project documentation.

## Usage
### Parameters
-- HYPER PARAMETERS --

### Building and Training the Model

## Datasets
The project uses the MNIST and CIFAR10 datasets for training and testing. The MNIST dataset is a large database of handwritten digits commonly used for training various image processing systems. The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels. The CIFAR10 dataset consists of 60,000 32x32 color images in 10 different classes.

### Training

### Data Augmentation
-- AUGMENTATION USED --

## Results
-- RESULTS --