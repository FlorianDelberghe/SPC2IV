"""Loads DICOM images and saves to png for GAN training with different contrasts"""

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
    # Saves one image in x
    one_in_x = 1

    for patient in patients:

        spc_volume = load_dcm_serie(patient['spc'])
        iv_volume = load_dcm_serie(patient['iv'])

        spc_volume = sitk.GetArrayFromImage(spc_volume)
        iv_volume = sitk.GetArrayFromImage(iv_volume)

        spc_volume_cont = tissue_contrast(spc_volume, rescale_intercept=0., method='sigmoid', contrast=['lung', 'soft', 'bone'])
        iv_volume_cont = tissue_contrast(iv_volume, rescale_intercept=0., method='sigmoid', contrast=['lung', 'soft', 'bone'])

        save_volume(spc_volume_cont[::one_in_x], "{}/{}/SPC".format(str(OUT_PATH), patient['id']), "{}_SPC.png".format(patient['id']), bit_depth=8)
        save_volume(iv_volume_cont[::one_in_x], "{}/{}/IV".format(str(OUT_PATH), patient['id']), "{}_IV.png".format(patient['id']), bit_depth=8)


if __name__ == '__main__':
    main()
