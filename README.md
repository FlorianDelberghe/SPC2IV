# CT CONSTRAST MODULATION SPC<=>IV  

Implementation of a CycleGAN architecture for CT contrast modulation.
The aim of the project was to artificially predict some of the CT contrast volumes from other contrast of the same volume.

### How to run code:

Create a path for the data:
`mkdir data/dicom` for GAN input as DICOM series, otherwise with .png: `mkdir data/images` then copy each patient as a different folder in the data_path.
Each folder in turn contains each contrast in a different folder with 'IV', 'SPC', 'ART' or 'TARD' in the folder name for the data_loader to find them (case insensitive)


Training the model
```
python train_models.py
```

testing the models output
```
python test_models.py
```

If you do not want to use DICOM images for concerns of speed, run:

```
python dicom2png.py <data_path> <output_path>
```

### Required  packages:  
tensorflow-gpu  
keras  
SimpleITK  
skimage
skimage.external.tifffile  
nibabel  
pydicom
pystackreg  
scipy
matplotlib