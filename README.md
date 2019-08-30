# CT CONSTRAST MODULATION SPC<=>IV  

Implementation of a CycleGAN architecture for CT contrast modulation.

### Required  packages:  
tensorflow-gpu  
keras  
SimpleITK  
skimage.external.tiffile  
nibabel  
pystackreg  

### How to run code:

Create a path for the data:
`mkdir data/dicom` for GAN input as DICOM series, otherwise with .png: `mkdir data/images` then copy each patient as a different folder in the data_path.
Each folder in turn contains each contrast in a different folder with 'IV' or 'SPC' in the folder name for the data_loader to find them (case insensitive)


Training the model
```
python train_models.py
```

testing the models output
```
python test_models.py
```

