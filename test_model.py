import os
import shutil
import sys

import keras
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import skimage.external.tifffile as tifffile
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import code.processing as processing
import code.utilities as utilities
from code.data_loader import DataLoaderCT, DataLoaderDICOM
from code.models import BasicCycleGAN, CMapCycleGAN, PairedLossCycleGAN


DICOM_PATH = 'data/dicom'


def main():
    # Dynamically alocates VRAM
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Training parameters
    params = {
        # Image specifics
        'input_res': (512, 512), 'n_channels': 2, 'dataset_name': 'spc2iv', 'data_contrast': 'coreg_sig', 'load_from': 'DICOM',
        # GAN layers
        'g_filters': 48, 'd_filters': 64, 'D_loss': 'mse', 'D_loss_weight': 1.0, 'cycle_loss_weight': 10.0, 'id_loss_weight': 1.0, 'pair_loss_weight': 2.0,
        'architecture': 'UNet', 'out_activation': 'tanh', 'transformer_layers': 9, 'D_output': 'patch',
        # Optimizer
        'initial_lr': 0.0002, 'start_epoch': 0, 'optimizer_scheduler': 'linear', 'cutoff_epoch': 40, 'max_epochs': 100,
        # Save the model and predict images also deletes what was in the previous output folder
        'save_state': False
        }

    gan = BasicCycleGAN(**params)
    gan.load_weights('models/saved/BasicCycleGanepoch_115', load_D=False)    

    patients = utilities.load_patient_path(DICOM_PATH)

    for i, patient in enumerate(patients):
        if i >= 10: break

        spc_test, spc_reader = utilities.load_dcm_serie(patient['spc'], True)
        iv_test, iv_reader = utilities.load_dcm_serie(patient['iv'], True)

        # for key in spc_reader.GetMetaDataKeys(1):
        #     print(key, ': ', spc_reader.GetMetaData(1, key))

        # utilities.save2dicom(spc_test, spc_reader, 'figures/test_dicom/', 'test_dcm')

        # [z,x,y]
        # spc_test = np.swapaxes(sitk.GetArrayFromImage(spc_test), 1,2)[:-1]
        # iv_test = np.swapaxes(sitk.GetArrayFromImage(iv_test), 1,2)[:-1]

        spc_test = sitk.GetArrayFromImage(spc_test)[:-1]
        iv_test = sitk.GetArrayFromImage(iv_test)[:-1]

        # utilities.save2nifti((spc_test +1024), np.eye(4),'figures/', 'spc_test.nii')
        # utilities.save2nifti((iv_test +1024), np.eye(4), 'figures/', 'iv_test.nii')

        spc_test_sig = processing.tissue_contrast(spc_test, rescale_intercept=-0, method='sigmoid', contrast=['soft', 'bone']) *2. -1.
        iv_test_sig = processing.tissue_contrast(iv_test, rescale_intercept=-0, method='sigmoid', contrast=['soft', 'bone']) *2. -1.
        
        # tifffile.imsave('figures/spc_test_sig.tif', spc_test[...,0].astype('float32'))
        # tifffile.imsave('figures/iv_test_sig.tif', iv_test[...,0].astype('float32'))

        iv_pred = (gan.predict_volume(spc_test_sig, 'spc') +1) /2.
        spc_pred = (gan.predict_volume(iv_test_sig, 'iv') +1) /2.

        # tifffile.imsave('figures/spc_pred_sig.tif', spc_pred[...,0].astype('float32'))
        # tifffile.imsave('figures/iv_pred_sig.tif', iv_pred[...,0].astype('float32'))

        spc_pred = processing.tissue_contrast(spc_pred, rescale_intercept=-1024, method='inv_sigmoid', contrast=['soft', 'bone'])
        iv_pred = processing.tissue_contrast(iv_pred, rescale_intercept=-1024, method='inv_sigmoid', contrast=['soft', 'bone'])

        spc_pred[spc_pred < 0] = 0
        iv_pred[iv_pred < 0] = 0

        # Removes noise in empty regions of the input image
        spc_pred[iv_test < -800] = 0
        iv_pred[spc_test < -800] = 0

        spc_pred = spc_pred.mean(axis=3)
        iv_pred = iv_pred.mean(axis=3)

        utilities.save2nifti(spc_pred, np.eye(4), 'figures/', 'spc_pred_{}.nii'.format(patient['id']))
        utilities.save2nifti(iv_pred, np.eye(4), 'figures/', 'iv_pred_{}.nii'.format(patient['id']))


if __name__ == '__main__':
    main()
