"""Loads DICOM images and saves to .png for GAN training with different contrasts"""

import os
import pathlib
import sys

import SimpleITK as sitk

from code.processing import tissue_contrast
from code.utilities import load_patient_path, save_volume, create_dir, load_dcm_serie


def main():

    DATA_PATH = pathlib.Path(sys.argv[1])
    OUT_PATH = pathlib.Path(sys.argv[2])

    if not DATA_PATH.exists():
        raise IOError("Path for the data: {!r} does not exists".format(str(DATA_PATH)))

    if not OUT_PATH.exists():
        create_dir(OUT_PATH)

    patients = load_patient_path(DATA_PATH)

    # Saves one slice every sampling interval
    sampling = 1
    contrasts = ['spc', 'art', 'iv', 'tard']

    for patient in patients:
        for cont in contrasts:
            
            c_volume = load_dcm_serie(patient[cont.lower()])
            c_volume = sitk.GetArrayFromImage(c_volume)

            c_volume_cont = tissue_contrast(c_volume, rescale_intercept=0., method='sigmoid', contrast=['lung', 'soft', 'bone'])

            save_volume(c_volume_cont[::sampling], "{}/{}/{}".format(str(OUT_PATH), patient['id'], cont.upper()),
                        "{}_{}.png".format(patient['id'], cont.upper()), bit_depth=8)


if __name__ == '__main__':
    main()
