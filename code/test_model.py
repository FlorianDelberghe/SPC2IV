import os
import shutil
import sys

import keras
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import skimage.external.tifffile as tifffile
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import SimpleITK as sitk

import models
import processing, utilities
from data_loader import DataLoaderCT, DataLoaderDICOM
from models import BasicCycleGAN, CMapCycleGAN, PairedLossCycleGAN

DICOM_PATH = '../data/dicom'


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
        # GAN layers
        'g_filters': 48, 'd_filters': 64, 'D_loss': 'mse', 'D_loss_weight': 1.0, 'cycle_loss_weight': 10.0, 'id_loss_weight': 1.0, 'pair_loss_weight': 2.0,
        'architecture': 'UNet', 'out_activation': 'tanh', 'transformer_layers': 9, 'D_output': 'patch',
        # Optimizer
        'initial_lr': 0.0002, 'start_epoch': 0, 'optimizer_scheduler': 'linear', 'cutoff_epoch': 40, 'max_epochs': 100,
        # Save the model and predict images also deletes what was in the previous output folder
        'save_state': False
        }

    gan = BasicCycleGAN(**params)
    gan.load_weights('models/spc2iv/BasicCycleGanepoch_110', load_D=False)    

    patients = utilities.load_patient_path(DICOM_PATH)

    # data_loader = DataLoaderCT('../data/images/coreg_sig', image_res=(512, 512), n_channels=2)
    # spc_test = gan.data_loader.load_test('spc')
    # iv_test = gan.data_loader.load_test('iv')


    patient = patients[4]

    spc_test, spc_reader = utilities.load_dcm_serie(patient['spc'], True)
    iv_test, iv_reader = utilities.load_dcm_serie(patient['iv'], True)

    # [z,x,y]
    spc_test = np.swapaxes(sitk.GetArrayFromImage(spc_test), 1,2)[:-1]
    iv_test = np.swapaxes(sitk.GetArrayFromImage(iv_test), 1,2)[:-1]

    utilities.save2nifti((spc_test +1024), np.eye(4),'figures/', 'spc_test.nii')
    utilities.save2nifti((iv_test +1024), np.eye(4), 'figures/', 'iv_test.nii')

    spc_test = processing.tissue_contrast(spc_test, rescale_intercept=-0, method='sigmoid', contrast=['soft', 'bone']) *2. -1.
    iv_test = processing.tissue_contrast(iv_test, rescale_intercept=-0, method='sigmoid', contrast=['soft', 'bone']) *2. -1.
    
    # tifffile.imsave('figures/spc_test_sig.tif', spc_test[...,0].astype('float32'))
    # tifffile.imsave('figures/iv_test_sig.tif', iv_test[...,0].astype('float32'))

    iv_pred = (gan.predict_volume(spc_test, 'spc') +1) /2.
    spc_pred = (gan.predict_volume(iv_test, 'iv') +1) /2.

    tifffile.imsave('figures/spc_pred_sig.tif', spc_pred[...,0].astype('float32'))
    tifffile.imsave('figures/iv_pred_sig.tif', iv_pred[...,0].astype('float32'))

    spc_pred = processing.tissue_contrast(spc_pred, rescale_intercept=-1024, method='inv_sigmoid', contrast=['soft', 'bone'])
    iv_pred = processing.tissue_contrast(iv_pred, rescale_intercept=-1024, method='inv_sigmoid', contrast=['soft', 'bone'])

    spc_pred[spc_pred < 0] = 0
    iv_pred[iv_pred < 0] = 0

    spc_pred = spc_pred.mean(axis=3)
    iv_pred = iv_pred.mean(axis=3)

    utilities.save2nifti(spc_pred, np.eye(4), 'figures/', 'spc_pred.nii')
    utilities.save2nifti(iv_pred, np.eye(4), 'figures/', 'iv_pred.nii')


if __name__ == '__main__':
    main()
