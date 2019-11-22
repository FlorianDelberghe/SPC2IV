import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import skimage.external.tifffile as tifffile
import skimage.io

from code.processing import tissue_contrast


def create_dir(path):
    """Creates destination folder if it does not already exists"""

    if not os.path.exists(path):
        print('Creating dir: {}'.format(path), end=' '*5)
        try:
            os.makedirs(path, exist_ok=True)
            print("Successful!")
        except:
            print("Failed")


def load_patient_path(data_path):
    """Loads all the patients folders with id and contrast paths
        PARAMS:
            data_path (str): path to the root of the patient files containing the dicom series

        RETURNS:
            patients (list(dict)): a list of dictionaries (one for every found patient) containing info such as serie_paths and patient_id"""

    print("Loading patients' data paths...")
    patients = []
    patient_folders = (folder for folder in os.listdir(data_path) if not '.' in folder)    

    for patient_folder in patient_folders:

        patient = {}
        patient['id'] = patient_folder
        patient['path'] = os.path.join(data_path, patient_folder)

        patient_contrasts = (folder for folder in os.listdir(os.path.join(data_path, patient_folder)))

        for patient_contrast in patient_contrasts:
            # Gets the path for the different contrasts
            if 'spc' in patient_contrast.lower() or 'sans iv' in patient_contrast.lower():
                patient['spc'] = os.path.join(data_path, patient_folder, patient_contrast)
            elif 'art' in patient_contrast.lower():
                patient['art'] = os.path.join(data_path, patient_folder, patient_contrast)
            elif 'iv' in patient_contrast.lower() and not 'sans' in patient_contrast.lower():
                patient['iv'] = os.path.join(data_path, patient_folder, patient_contrast)
            elif 'tard' in patient_contrast.lower():
                patient['tard'] = os.path.join(data_path, patient_folder, patient_contrast)

        patients.append(patient)

    return patients


def load_dcm_meta(filepath, contrast):
    """Loads usefull metadata from a dicom image, builds image2world affine transform"""

    ds = pydicom.dcmread(filepath)
    metadata = {}

    d_x, d_y = ds.PixelSpacing
    orient = np.array(ds.ImageOrientationPatient).reshape((2,3)).T

    affine_transform = np.zeros((4,4), dtype=np.float32)
    affine_transform[:,-1] = [*ds.ImagePositionPatient, 1]
    affine_transform[0:-1, 0:2] = np.repeat(np.array([[d_x, d_y]]), 3, axis=0) *orient
    affine_transform[2, 2] = ds.SliceThickness
    
    metadata['contrast'] = contrast.upper()
    metadata['affine_transform'] = affine_transform

    # Sets the other error prone metadata with 'error handling'
    for key, value in zip(['rescale_intercept', 'rescale_slope', 'id'], 
                          ['ds.RescaleIntercept', 'ds.RescaleSlope', 'ds.PatientID']):
        try:
            metadata[key] = eval(value)
        except AttributeError as e:
            print(str(e))
            pass

    return metadata


def load_img_volumes(patient_dict, contrast=['spc', 'art', 'iv', 'tard']):
    """Loads all the dicom images for the 4 contrasts in the patient's folder (Deprecated, use load_dcm_serie() instead)"""

    contrast_vol = []

    for c in contrast:
        if c in patient_dict.keys():

            c_vol = []
            dcm_files = [file for file in os.listdir(patient_dict[c]) if file.endswith('.dcm')]
            dcm_files.sort()

            for i, file in enumerate(dcm_files):
                # Discards the first image of the serie
                if i == 0: continue          

                elif i == 1:
                    
                    current_serie = file[3:7]
                    #Loading serie metadata
                    
                    metadata = load_dcm_meta(os.path.join(patient_dict[c], file), c)
                    metadata.setdefault('id', patient_dict['id'])

                # Each folder should only contain 1 serie of images 2 means that there is a duplicate
                elif file[3:7] != current_serie:
                    print(r"/!\ Duplicate serie", end='')
                    break

                print("\rLoading contrast {} for patient {} slice: {}".format(c, patient_dict['id'], i), end=' '*10)

                try:
                    ds = pydicom.dcmread(os.path.join(patient_dict[c], file))
                    c_vol.append(ds.pixel_array)

                # Weird error in pydicom when importing certain images
                except NotImplementedError:
                    print("<- Falied to load file: '{}'".format(file))
                    #TODO get shape from the dicom metadata
                    c_vol.append(np.full(c_vol[0].shape, -1))

        contrast_vol.append({'img_volume': np.array(c_vol), 'metadata': metadata})
        print('')

    return contrast_vol


def save_volume(volume, folder_path, filename, axis=0, bit_depth=8, **kwargs):
    """"Saves image volume to requiered format
        PARAMS:
            volume (np.array): image volume to save
            folder_path (str): folder in which to save the image
            filename (str): name of the file to be saved (acts as prefix for 'png' format)
            axis (int): axis along which to slice
            bit_depth (int): bit depth of the saved images supported: {8, 16, 32} 

            file_format (str): optional when not specified implicitely by filename"""

    def rescale(image, bit_depth):
        """Rescales pixel values in the correct range to save as images
            PARAMS:
                image (np.array): array of positive values for bit_depth={8, 16}
                
                bit_depth (int): bit depth to return the image and deduce rescaling behavior
            RETURNS:
                (np.array): rescaled changed dtype image"""

        if bit_depth == 32:
            # Rescales image range to [0,1] float32
            image -= image.min()
            image *= 1./image.max()
            return image.astype('float32')

        elif bit_depth == 8:
            array_dtype = 'uint8'
            image = np.squeeze(image)

            # Rescales image from [min, max] -> [0, 255]
            if (image < 0).any():
                print(f"Min of img is: {iamge.min()}, clipping pixel values")
                image[image < 0] = 0
           
            # One channel image
            if len(image.shape) == 2:                    
                return (image /image.max() *(2**bit_depth -1)).astype(array_dtype)

            # Multi channels image
            elif len(image.shape) == 3 and volume.shape[-1] == 3:                    
                norm_channels = [(image[...,i] /image[...,i].max() *(2**bit_depth -1)).astype(array_dtype) for i in range(image.shape[-1])]
                return np.stack(norm_channels, axis=-1)

            else:
                raise ValueError("Dimension error for volume n_channel(s): {}".format(image.shape[-1]))

        elif bit_depth == 16:
            # Clips weights outside of range [uint16_min, int16_max] return unchanged pixel values otherwise
            image[image < 0] = 0
            image[image >  2**(bit_depth-1) -1] = 2**(bit_depth-1) -1
            return image.astype('uint16')

        else:
            raise ValueError(f"Wrong value for bit_depth: {bit_depth} should be 8, 16, 32")

    # Finds the wanted saving format
    try:
        file_format = filename.split('.')[-1] if filename.split('.')[-1] in ['png', 'tif', 'pkl'] else kwargs['file_format']
    except KeyError:
        print("File format could not be infered from filename or 'file_format' kwarg")
        raise

    create_dir(folder_path)

    # Saves volume to serie of .png images in a new folder
    if file_format == 'png':
        slc = [slice(None)] *len(volume.shape)
        for i in range(volume.shape[axis]):
            slc[axis] = slice(i,i+1)
            print("\rSaving: {}_{}.png".format(filename.split('.')[0], i+1), end=' '*10)
            tifffile.imsave(os.path.join(folder_path, "{}_{}.png".format(filename.split('.')[0], i+1)), rescale(volume[tuple(slc)], bit_depth))
        print('')

    # Saves volume to tif
    elif file_format == 'tif':
        #TODO change main axis
        print("Saving: {}".format(os.path.join(folder_path, filename)))
        volume = np.swapaxes(volume[:, ::-1,::-1], 1,2)
        tifffile.imsave(os.path.join(folder_path, filename), rescale(volume, bit_depth))

    elif file_format == 'pkl':
        with open(os.path.join(folder_path, "{}.pkl".format(filename.split('.')[0])), mode='wb') as f:
            print("Saving as '{}' to '{}'".format("{}.pkl".format(filename), folder_path))
            pickle.dump(volume, f)

    else:
        raise ValueError("Unsupported file format: {}".format(file_format))


def load_dcm_serie(serie_path, return_reader=False):
    """Loads a DICOM serie using SimpleITK
        PARAMS: 
            serie_path (str): path to the requiered serie
            return_reader (bool): wheter to return the ImageSeriesReader or not
            
        RETURNS:
            dcm_serie (sitk.Image): sitk 3D image in real Houndsfield units with dims = [x,y,z] 
            reader (sitk.ImageSeriesReader): used to get additional metadata for the whole serie"""

    reader = sitk.ImageSeriesReader()
    # Reads serie files
    dicom_names = reader.GetGDCMSeriesFileNames(serie_path)
    reader.SetFileNames(dicom_names)

    # MetaData laoding params
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    print(f"Loading serie at: {serie_path}")
    dcm_serie = reader.Execute()
    # Sets the spacing with the 'SliceThickness' MetaData as it can be wrongfully initialized (read metadeta on slice 1!)
    dcm_serie.SetSpacing((*dcm_serie.GetSpacing()[:2], float(reader.GetMetaData(1, '0018|0050'))))

    if return_reader:
        return dcm_serie, reader
    else:
        return dcm_serie


def save2dicom(volume, reader, serie_description, folder_path, fileprefix):
    """Saves sitk image volume to dicom
        PARAMS:
            volume (sitk image): Input volume as sitk image
            reader (sitk image reader): reader to get the metadatafrom
            serie_description ():
            folder_path (str): folder where to store the files
            fileprefix (str): filename to be followed by '_(n_slice)'"""

    create_dir(folder_path)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are also accessible).
    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
                    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = volume.GetDirection()
    series_tag_values = [(k, reader.GetMetaData(0,k)) for k in tags_to_copy if reader.HasMetaDataKey(0,k)] + \
        [("0008|0031", modification_time),  # Series Time
         ("0008|0021", modification_date),  # Series Date
         ("0008|0008", "DERIVED\\SECONDARY\\GANPRED"),  # Image Type
         ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date + \
          ".1"+modification_time),  # Series Instance UID
         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                           direction[1], direction[4], direction[7])))),
         ("0008|103e", serie_description),  # Series Description
         ("0018|0050", str(volume.GetSpacing()[-1])), # Slice Thickness
         ("0028|1052", str(-1024)), # Rescale Intercept
         ("0028|1053", str(1.0)) # Rescale Slope
         ]

    print(f"Saving to {folder_path}")
    for i in range(volume.GetDepth()):
        image_slice = volume[:,:,i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)

        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,volume.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join("{}/{}_{}.dcm".format(folder_path, fileprefix, str(i))))
        writer.Execute(image_slice)


def save2nifti(volume, affine_mat, out_folder, filename):
    """Saves volume to nifti format"""

    # Clipping the pixels values to positive int16 range
    volume[volume < 0] = 0
    volume[volume > 2**15-1] = 2**15-1

    # Axis in the right order for nifti format
    volume = np.swapaxes(volume, 0, 2) #[z,y,x] -> [x,y,z]
    volume = volume[::-1,::-1] #[x,y,z] -> [-x,-y,z]

    # Converting to NIFTI and saving
    volume_nifti = nib.Nifti1Image(volume.astype('uint16'), affine_mat)
    create_dir(out_folder)
    print("Saving to : {}".format(os.path.join(out_folder, filename)))
    volume_nifti.to_filename(os.path.join(out_folder, filename))
