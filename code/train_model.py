import os
import shutil
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import skimage.external.tifffile as tifffile
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import models
from data_loader import DataLoaderCT
from models import BasicCycleGAN, CMapCycleGAN, PairedLossCycleGAN, ResCycleGAN


def main():

    # Dynamically alocates VRAM
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Training parameters
    params = {
        # Image specifics
        'input_res': (512, 512), 'n_channels': 2, 'dataset_name': 'spc2iv', 'data_contrast': 'coreg_sig',
        # GAN training
        'D_loss_weight': 2.0, 'cycle_loss_weight': 10.0, 'id_loss_weight': 1.0, 'pair_loss_weight': 2.0, 'normalization': 'batch',
        # Generator
        'g_filters': 48, 'architecture': 'unet', 'out_activation': 'tanh', 'transformer_layers': 9, 
        # DIcriminator
        'd_filters': 64, 'D_output': 'patch', 'D_loss': 'mse', 'D_out_activation': 'sigmoid',
        # Optimizer
        'initial_lr': 0.0002, 'start_epoch': 0, 'optimizer_scheduler': 'linear', 'cutoff_epoch': 50, 'max_epochs': 120,
        # Save the model and predict images also deletes what was in the previous output folder
        'save_state': True
        }

    gan = BasicCycleGAN(**params)
    print(gan)

    # gan.load_weights('models/saved/epoch_30_AE_noncat', load_D=True)
    gan.train(epochs=params['max_epochs'], batch_size=2, sample_interval=200, starting_epoch=0)


if __name__ == '__main__':
    main()
