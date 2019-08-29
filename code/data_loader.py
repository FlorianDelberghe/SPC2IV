import os
from glob import glob
from random import sample

import numpy as np
import scipy
import SimpleITK as sitk
import skimage.external.tifffile as tifffile

import processing, utilities


class DataLoader():

    def __init__(self, name='defaultDataLoader'):
        self.name = name


    def __repr__(self):
        return "{}".format(self.name)


    def __str__(self):
        return "{} loads from {}".format(self.name, self.data_path)


class DataLoaderCT(DataLoader):

    def __init__(self, data_path, image_res=(512, 512), n_channels=1, one_in_x=10):
        super(DataLoaderCT, self).__init__('DataLoaderCT')

        self.data_path = data_path
        self.img_res = image_res
        self.n_channels = n_channels
        self.one_in_x = one_in_x
        
        # Sets one of the folder as the testing dataset
        self.patients = [p for p in os.listdir(self.data_path) if not p.startswith('.')]
        self.test_patient = sample(self.patients, 1)
        self.patients.remove(self.test_patient[0])

        # Chooses the validation dataset and loads the filepaths
        self.val_patient = None
        self.update_val()

    
    def load_patient_files(self):
        """Loads the path to the training and validation datasets"""

        def load_folder(folder):
            return [os.path.join(self.data_path, p, folder, filename) 
                    for filename in os.listdir(os.path.join(self.data_path, p, folder))
                    if not (filename.startswith('.') or filename.endswith('.json'))]

        # Sets training filepaths
        self.tr_spc_files, self.tr_iv_files = [], []
        for p in self.patients:
            spc_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                          if 'SPC' in folder][0]
            iv_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                         if 'iv' in folder or 'IV' in folder][0]

            self.tr_spc_files.extend(load_folder(spc_folder))
            self.tr_iv_files.extend(load_folder(iv_folder))

        self.tr_spc_files.sort()
        self.tr_iv_files.sort()
        self.tr_spc_files, self.tr_iv_files = np.array(self.tr_spc_files), np.array(self.tr_iv_files)

        # Sets validation filepaths
        self.val_spc_files, self.val_iv_files = [], []
        for p in self.val_patient:
            spc_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                          if 'SPC' in folder][0]
            iv_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p)) 
                         if 'iv' in folder or 'IV' in folder][0]

            self.val_spc_files.extend(load_folder(spc_folder))
            self.val_iv_files.extend(load_folder(iv_folder))

        self.val_spc_files.sort()
        self.val_iv_files.sort()
        self.val_spc_files, self.val_iv_files = np.array(self.val_spc_files), np.array(self.val_iv_files)

        # Sets testing filepaths
        self.test_spc_files, self.test_iv_files = [], []
        for p in self.test_patient:
            spc_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                          if 'SPC' in folder][0]
            iv_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p)) 
                         if 'iv' in folder or 'IV' in folder][0]

            self.test_spc_files.extend(load_folder(spc_folder))
            self.test_iv_files.extend(load_folder(iv_folder))

        self.test_spc_files.sort()
        self.test_iv_files.sort()
        self.test_spc_files, self.val_iv_files = np.array(self.test_spc_files), np.array(self.test_iv_files)


    def update_val(self):
        """Changes the validation set of the data loader"""
        
        if self.val_patient is not None:
            self.patients.extend(self.val_patient)

        self.val_patient = sample(self.patients, 1)
        self.patients.remove(self.val_patient[0])

        # Updates the values for the training and validation file paths
        self.load_patient_files()


    def load_data(self, contrast, batch_size=1):
        """Loads batch_size images from the desired contrast of the available data"""

        if contrast.lower() == 'spc':
            data = [self.load_image(file) for file in self.val_spc_files[np.random.choice(len(self.val_spc_files), batch_size)]]
        elif contrast.lower() == 'iv':
            data = [self.load_image(file) for file in self.val_iv_files[np.random.choice(len(self.val_iv_files), batch_size)]]
        else:
            raise ValueError("Wrong contrast: {}".format(contrast))

        return np.array(data)


    def load_batch(self, batch_size=1, paired=True):
        """Creates a generator that yields a single batch of the data for both contrasts"""

        self.n_batches = int(min(len(self.tr_spc_files), len(self.tr_iv_files)) /batch_size)
        total_samples = self.n_batches * batch_size

        spc_ind = np.random.choice(len(self.tr_spc_files), total_samples)
        iv_ind = np.random.choice(len(self.tr_iv_files), total_samples)

        for i in range(0, self.n_batches, self.one_in_x):
            spc_imgs = [self.load_image(file) for file in self.tr_spc_files[spc_ind[i *batch_size:(i+1) *batch_size]]]
            iv_paired = [self.load_image(file) for file in self.tr_iv_files[spc_ind[i *batch_size:(i+1) *batch_size]]]
            
            iv_imgs = [self.load_image(file) for file in self.tr_iv_files[iv_ind[i *batch_size:(i+1) *batch_size]]]
            spc_paired = [self.load_image(file) for file in self.tr_spc_files[iv_ind[i *batch_size:(i+1) *batch_size]]]

            if paired:
                yield np.array(spc_imgs), np.array(iv_imgs), np.array(iv_paired), np.array(spc_paired)
            else:
                yield np.array(spc_imgs), np.array(iv_imgs)


    def load_test(self, contrast, paired=False):

        if contrast.lower() == 'iv':
            data = [self.load_image(file) for file in self.test_iv_files]
        elif contrast.lower() == 'spc':
            data = [self.load_image(file) for file in self.test_spc_files]
        else:
            raise ValueError("Unknown contrast: {!r}".format(contrast))

        return np.stack(data, axis=0)
        


    def load_image(self, filepath, bit_depth=8):
        """Loads images from .png files
            -1 channel: soft tissue contrast only
            -2 channels: soft and bone tissue contrasts
            -3 channels: lung, soft and bone tissue contrasts
        """
        def normalize(image, bit_depth=8, rg=(-1, 1)):
            rg_width = float(rg[1] - rg[0])
            return image /(2**bit_depth -1) *rg_width +rg[0]

        if self.n_channels == 1:
            #Loads only the soft tissue contrast channel
            image = tifffile.imread(filepath)
            if len(image.shape) == 3:
                return normalize(np.expand_dims(image[...,1], axis=-1), bit_depth)
            else:
                return normalize(np.expand_dims(image, axis=-1), bit_depth)

        elif self.n_channels == 2:
            return normalize(tifffile.imread(filepath)[..., 1:], bit_depth)
        elif self.n_channels == 3:
            return normalize(tifffile.imread(filepath), bit_depth)
        else:
            raise ValueError("Wrong number of channels '{}'".format(self.n_channels))

#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

class DataLoaderDICOM(DataLoader):

    def __init__(self, data_path, image_res=(512, 512), channels=['soft'], windowing_method='sigmoid', one_in_x=10):
        super(DataLoaderDICOM, self).__init__('DataLoaderDICOM')

        self.data_path = data_path
        self.img_res = image_res
        self.channels = channels
        self.n_channels = len(channels)
        self.one_in_x = one_in_x
        self.windowing_method = windowing_method
        
        # Sets one of the folder as the testing dataset
        self.patient_list = utilities.load_patient_path(self.data_path)
        self.test_patient = sample(self.patient_list, 1)
        self.patient_list.remove(self.test_patient[0])

        # Chooses the validation dataset and loads the filepaths
        self.val_patient = None
        self.update_val()


    def load_patient_files(self):
        """Loads the path to the training and validation datasets"""

        def load_folder(folder):
            return [os.path.join(folder, filename) for filename in os.listdir(os.path.join(folder))
                    if filename.endswith('.dcm')]

        # Sets testing filepaths
        self.test_spc_files, self.test_iv_files = [], []
        for patient in self.test_patient:
            spc_folder = patient['spc']
            iv_folder = patient['iv'] 

            self.test_spc_files.extend(load_folder(spc_folder))
            self.test_iv_files.extend(load_folder(iv_folder))

        self.test_spc_files.sort()
        self.test_iv_files.sort()
        self.test_spc_files, self.val_iv_files = np.array(self.test_spc_files), np.array(self.test_iv_files)

        # Sets validation filepaths
        self.val_spc_files, self.val_iv_files = [], []
        for patient in self.val_patient:
            spc_folder = patient['spc']
            iv_folder = patient['iv'] 

            self.val_spc_files.extend(load_folder(spc_folder))
            self.val_iv_files.extend(load_folder(iv_folder))

        self.val_spc_files.sort()
        self.val_iv_files.sort()
        self.val_spc_files, self.val_iv_files = np.array(self.val_spc_files), np.array(self.val_iv_files)

        # Sets training filepaths
        self.tr_spc_files, self.tr_iv_files = [], []
        for patient in self.patient_list:
            spc_folder = patient['spc']
            iv_folder = patient['iv'] 

            self.tr_spc_files.extend(load_folder(spc_folder))
            self.tr_iv_files.extend(load_folder(iv_folder))

        self.tr_spc_files.sort()
        self.tr_iv_files.sort()
        self.tr_spc_files, self.tr_iv_files = np.array(self.tr_spc_files), np.array(self.tr_iv_files)


    def update_val(self):
        """Changes the validation set of the data loader"""
        
        if self.val_patient is not None:
            self.patient_list.extend(self.val_patient)

        self.val_patient = sample(self.patient_list, 1)
        self.patient_list.remove(self.val_patient[0])

        # Updates the values for the training and validation file paths
        self.load_patient_files()


    def load_data(self, contrast, batch_size=1):
        """Loads batch_size images from the desired contrast of the available validation data"""

        if contrast.lower() == 'spc':
            data = [self.load_image(file) for file in self.val_spc_files[np.random.choice(len(self.val_spc_files), batch_size)]]
        elif contrast.lower() == 'iv':
            data = [self.load_image(file) for file in self.val_iv_files[np.random.choice(len(self.val_iv_files), batch_size)]]
        else:
            raise ValueError("Wrong contrast: {}".format(contrast))

        return np.concatenate(data, axis=0)


    def load_batch(self, batch_size=1):
        """Creates a generator that yields a single batch of the data for both contrasts"""

        self.n_batches = int(min(len(self.tr_spc_files), len(self.tr_iv_files)) /batch_size)
        print(len(self.tr_spc_files), len(self.tr_iv_files),self.n_batches)
        total_samples = self.n_batches * batch_size

        spc_ind = np.random.choice(len(self.tr_spc_files), total_samples)
        iv_ind = np.random.choice(len(self.tr_iv_files), total_samples)

        for i in range(0, self.n_batches, self.one_in_x):
            spc_imgs = [self.load_image(file) for file in self.tr_spc_files[spc_ind[i *batch_size:(i+1) *batch_size]]]            
            iv_imgs = [self.load_image(file) for file in self.tr_iv_files[iv_ind[i *batch_size:(i+1) *batch_size]]]

            yield np.concatenate(spc_imgs, axis=0), np.concatenate(iv_imgs, axis=0)


    def load_image(self, filepath):
        """Loads images from .dcm files
        """

        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        image = reader.Execute()
        
        image = np.swapaxes(sitk.GetArrayFromImage(image), 1,2)        
        # NO need for rescale_intercept when loading with sitk
        image = processing.tissue_contrast(image, rescale_intercept=-0, rescale_slope=1., 
                                           method=self.windowing_method, contrast=self.channels)

        # Rescales [0,1] -> [-1,1]
        return image *2. -1.
