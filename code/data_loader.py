import os
from random import sample

import numpy as np
import scipy
import SimpleITK as sitk
import skimage.external.tifffile as tifffile

import code.processing as processing
import code.utilities as utilities


class DataLoader():
    """Mother class for the different DataLoaders"""
    
    def __init__(self, name='defaultDataLoader'):
        self.name = name


    def __repr__(self):
        return f"{self.name}"


    def __str__(self):
        return self.__repr__()


class DataLoaderCT(DataLoader):
    """GAN data_loader loading data from pre-processed image volumes saved to .png slices"""

    def __init__(self, data_path, contrasts=['spc', 'iv'], image_res=(512, 512), n_channels=1, sampling=10):
        super(DataLoaderCT, self).__init__('DataLoaderCT')

        self.data_path = data_path
        self.contrasts = contrasts
        self.img_res = image_res
        self.n_channels = n_channels
        self.sampling = sampling
        
        # Sets one of the folder as the testing dataset
        self.patients = [p for p in os.listdir(self.data_path) if not p.startswith('.')]
        self.test_patient = sample(self.patients, 1)
        self.patients.remove(self.test_patient[0])

        # Chooses the validation dataset and loads the filepaths
        self.val_patient = None
        self.update_val()


    def __repr__(self):
        return "{} loads {} from {}".format(self.name, self.contrasts, self.data_path)


    def __str__(self):
        super(DataLoaderCT, self).__str__()

    
    def load_patient_files(self):
        """Loads the path to the training and validation datasets"""

        def load_folder(folder):

            return [os.path.join(self.data_path, p, folder, filename) 
                    for filename in os.listdir(os.path.join(self.data_path, p, folder))
                    if not (filename.startswith('.') or filename.endswith('.json'))]

        # Sets training filepaths
        self.tr_A_files, self.tr_B_files = [], []
        for p in self.patients:
            A_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[0].lower() in folder.lower()][0]
            B_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[1].lower() in folder.lower()][0]

            self.tr_A_files.extend(load_folder(A_folder))
            self.tr_B_files.extend(load_folder(B_folder))

        # Keeps the order of the files (usefull for paired loading)
        self.tr_A_files.sort()
        self.tr_B_files.sort()
        self.tr_A_files, self.tr_B_files = np.array(self.tr_A_files), np.array(self.tr_B_files)

        # Sets validation filepaths
        self.val_A_files, self.val_B_files = [], []
        for p in self.val_patient:
            A_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[0].lower() in folder.lower()][0]
            B_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[1].lower() in folder.lower()][0]

            self.val_A_files.extend(load_folder(A_folder))
            self.val_B_files.extend(load_folder(B_folder))

        self.val_A_files.sort()
        self.val_B_files.sort()
        self.val_A_files, self.val_B_files = np.array(self.val_A_files), np.array(self.val_B_files)

        # Sets testing filepaths
        self.test_A_files, self.test_B_files = [], []
        for p in self.test_patient:
            A_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[0].lower() in folder.lower()][0]
            B_folder = [folder for folder in os.listdir(os.path.join(self.data_path, p))
                        if self.contrasts[1].lower() in folder.lower()][0]

            self.test_A_files.extend(load_folder(A_folder))
            self.test_B_files.extend(load_folder(B_folder))

        self.test_A_files.sort()
        self.test_B_files.sort()
        self.test_A_files, self.val_B_files = np.array(self.test_A_files), np.array(self.test_B_files)


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

        if contrast.lower() == self.contrasts[0].lower():
            data = [self.load_image(file) for file in self.val_A_files[np.random.choice(len(self.val_A_files), batch_size)]]

        elif contrast.lower() == self.contrasts[1].lower():
            data = [self.load_image(file) for file in self.val_B_files[np.random.choice(len(self.val_B_files), batch_size)]]

        else:
            raise ValueError(f"Wrong contrast: {contrast}")

        return np.array(data)


    def load_batch(self, batch_size=1, paired=True):
        """Creates a generator that yields a single batch of the data for both contrasts"""

        # Max available batches
        self.n_batches = int(min(len(self.tr_A_files), len(self.tr_B_files)) /batch_size)
        total_samples = self.n_batches * batch_size

        A_ind = np.random.choice(len(self.tr_A_files), total_samples)
        B_ind = np.random.choice(len(self.tr_B_files), total_samples)

        for i in range(0, self.n_batches, self.sampling):

            A_imgs = [self.load_image(file) for file in self.tr_A_files[A_ind[i *batch_size:(i+1) *batch_size]]]
            B_imgs = [self.load_image(file) for file in self.tr_B_files[B_ind[i *batch_size:(i+1) *batch_size]]]

            if not paired:
                yield np.array(A_imgs), np.array(B_imgs)

            else:
                B_paired = [self.load_image(file) for file in self.tr_B_files[A_ind[i *batch_size:(i+1) *batch_size]]]            
                A_paired = [self.load_image(file) for file in self.tr_A_files[B_ind[i *batch_size:(i+1) *batch_size]]]

                yield np.array(A_imgs), np.array(B_imgs), np.array(B_paired), np.array(A_paired)


    def load_test(self, contrast, paired=False):
        """Loads images from the test set"""

        if contrast.lower() == self.contrasts[0].lower():
            data = [self.load_image(file) for file in self.test_A_files]

        elif contrast.lower() == self.contrasts[1].lower():
            data = [self.load_image(file) for file in self.test_B_files]

        else:
            raise ValueError(f"Unknown contrast: {contrast!r}")

        return np.stack(data, axis=0)        


    def load_image(self, filepath, bit_depth=8):
        """Loads images from .png files
            -1 channel: soft tissue contrast only
            -2 channels: soft and bone tissue contrasts
            -3 channels: lung, soft and bone tissue contrasts"""

        def normalize(image, bit_depth=8, rg=(-1, 1)):
            """Linearly rescales data to the wanted range (rg)"""

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
            return normalize(tifffile.imread(filepath)[...,1:], bit_depth)

        elif self.n_channels == 3:
            return normalize(tifffile.imread(filepath), bit_depth)

        else:
            raise ValueError(f"Wrong number of channels {self.n_channels}")

#------------------------------------------------------------#
#                                                            #
#                                                            #
#------------------------------------------------------------#

class DataLoaderDICOM(DataLoader):
    """GAN data_loader loading data from raw DICOM series slice by slice"""

    def __init__(self, data_path, image_res=(512, 512), contrasts=['spc', 'iv'], channels=['soft'], windowing_method='sigmoid', sampling=10, **kwargs):

        super(DataLoaderDICOM, self).__init__('DataLoaderDICOM')

        self.data_path = data_path
        self.contrasts = contrasts

        self.img_res = image_res
        self.channels = channels
        self.n_channels = len(channels)
        self.sampling =sampling
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
            spc_folder = patient[self.contrasts[0].lower()]
            iv_folder = patient[self.contrasts[1].lower()] 

            self.test_spc_files.extend(load_folder(spc_folder))
            self.test_iv_files.extend(load_folder(iv_folder))

        self.test_spc_files.sort()
        self.test_iv_files.sort()
        self.test_spc_files, self.val_iv_files = np.array(self.test_spc_files), np.array(self.test_iv_files)

        # Sets validation filepaths
        self.val_spc_files, self.val_iv_files = [], []
        for patient in self.val_patient:
            spc_folder = patient[self.contrasts[0].lower()]
            iv_folder = patient[self.contrasts[1].lower()] 

            self.val_spc_files.extend(load_folder(spc_folder))
            self.val_iv_files.extend(load_folder(iv_folder))

        self.val_spc_files.sort()
        self.val_iv_files.sort()
        self.val_spc_files, self.val_iv_files = np.array(self.val_spc_files), np.array(self.val_iv_files)

        # Sets training filepaths
        self.tr_spc_files, self.tr_iv_files = [], []
        for patient in self.patient_list:
            spc_folder = patient[self.contrasts[0].lower()]
            iv_folder = patient[self.contrasts[1].lower()] 

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

        if contrast.lower() == self.contrasts[0].lower():
            data = [self.load_image(file) for file in self.val_spc_files[np.random.choice(len(self.val_spc_files), batch_size)]]
        elif contrast.lower() == self.contrasts[1].lower():
            data = [self.load_image(file) for file in self.val_iv_files[np.random.choice(len(self.val_iv_files), batch_size)]]
        else:
            raise ValueError("Wrong contrast: {}".format(contrast))

        return np.concatenate(data, axis=0)


    def load_batch(self, batch_size=1):
        """Creates a generator that yields a single batch of the data for both contrasts"""

        self.n_batches = int(min(len(self.tr_spc_files), len(self.tr_iv_files)) /batch_size)
        total_samples = self.n_batches * batch_size

        spc_ind = np.random.choice(len(self.tr_spc_files), total_samples)
        iv_ind = np.random.choice(len(self.tr_iv_files), total_samples)

        for i in range(0, self.n_batches, self.sampling):
            spc_imgs = [self.load_image(file) for file in self.tr_spc_files[spc_ind[i *batch_size:(i+1) *batch_size]]]            
            iv_imgs = [self.load_image(file) for file in self.tr_iv_files[iv_ind[i *batch_size:(i+1) *batch_size]]]

            yield np.concatenate(spc_imgs, axis=0), np.concatenate(iv_imgs, axis=0)


    def load_image(self, filepath):
        """Loads images from .dcm files"""

        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        image = reader.Execute()
        
        # image = np.swapaxes(sitk.GetArrayFromImage(image), 1,2)        
        image = sitk.GetArrayFromImage(image)
        # NO need for rescale_intercept when loading with sitk
        image = processing.tissue_contrast(image, rescale_intercept=-0, rescale_slope=1., 
                                           method=self.windowing_method, contrast=self.channels)

        # Rescales [0,1] -> [-1,1]
        return image *2. -1.
