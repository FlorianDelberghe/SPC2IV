import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.external.tifffile as tifffile
from pystackreg import StackReg


def tissue_contrast(volume, rescale_intercept=-1024, rescale_slope=1., method='linear', contrast=['lung', 'soft', 'bone']):
    """Rescales contrast of volume to [0, 1] by windowing or reverses sigmoid windowing
        INPUTS:
            volume (np.array): input stack of images to rescale, dims don't matter for the windowing, for sigmoid recontruction last dim must be channels (contrasts)
                vol.shape = [x,y,z, channels]

            rescale intercept (int/float): Origin real value for recontruction of true Houndsfield units 

            rescale_slope (float): Linear slope coefficient for recontruction of true Houndsfield units

            method (str): contrast method for rescaling/reconstruction, must be in ['linear', 'sigmoid', 'inv_sigmoid']

            contrasts (list(str)): list of contrasts that will be conputed/reverted must contain: 'lung', 'soft', 'bone' exclusively
        
        OUTPUTS:
            conts/inv_conts (list(np.array)): list of contrast/restored images rescaled in the desirered windows"""

    def lin_rescale(volume, width, level):
        """Rescales volume pixel values linearly [level -width/2, level +width/2] -> [0, 1] """
        min_level, max_level = level -width/2, level +width/2
        # Linear rescaling fo pix values
        rescaled_volume = (volume-min_level) /(max_level-min_level)
        # Value clipping for extremas
        rescaled_volume[rescaled_volume < 0] = 0
        rescaled_volume[rescaled_volume > 1] = 1
        return rescaled_volume

    def sigmoid_rescale(volume, width, level):
        """Rescales volume pixel values with sigmoid around 'level' with window size proprotional to 'width' """
        return 1./(1.+np.exp(-(volume-level)/width))

    def inv_sig_rescale(volume, width, level):
        """Revertes the sigmoid rescaling given 'level' and 'width' """
        # float64 precision
        epsilon = np.finfo(np.float).eps
        # Shifts the extramas values {0, 1} by epsilon to avoid ZeroDivision and log(0)
        volume[volume <= 0] = epsilon
        volume[volume >= 1] = 1 -epsilon
        return np.around((-width *np.log(1./volume -1) +level), decimals=0)

    #(win_width, win_level)
    windows_lin = {'lung': (1800., -600.), 'soft': (400., 50.), 'bone': (1800., 400.)}
    windows_sig = {'lung': (200., -500.), 'soft': (80., 50.), 'bone': (150., 350.)}

    # New window params after trials
    windows_lin = {'lung': (560., 80.), 'soft': (570., 950.), 'bone': (400., 1100.)}
    # windows_sig = {'lung': (150., -650.), 'soft': (55., 45.), 'bone': (65., 200.)} 

    print("Rescaling image using {!r} method".format(method))

    if method == 'linear':
        #Rescaling to real Houndsfield Units
        volume = (volume +rescale_intercept).astype('float64') *float(rescale_slope)

        conts = []
        for cont in contrast:
            conts.append(lin_rescale(volume, windows_lin[cont][0], windows_lin[cont][1]))

    elif method == 'sigmoid':     
        #Rescaling to real Houndsfield Units
        volume = (volume +rescale_intercept).astype('float64') *float(rescale_slope) 

        conts = []
        for cont in contrast:
            conts.append(sigmoid_rescale(volume, windows_sig[cont][0], windows_sig[cont][1]))

    elif method == 'inv_sigmoid':
        inv_conts = []
        for i, c in enumerate(contrast):
            inv_cont = inv_sig_rescale(volume[...,i].astype('float64'), windows_sig[c][0], windows_sig[c][1])
            # Rescaling from Houndsfield Units to uint range /!\ Values may be <0 due to numerical imprecisions best to clip outside
            inv_cont /= float(rescale_slope)
            inv_cont -= rescale_intercept                
            inv_conts.append(inv_cont)

        return np.stack(inv_conts, axis=3)

    else:
        raise ValueError("'method' must be 'linear', 'sigmoid' or 'inv_sigmoid', is: {!r}".format(method))

    return np.stack(conts, axis=3)


def rough_coregister(ref_volume, realign_volume, affine_transform, axis=2):
    """Rapid coregistration method using PyStackReg package
        INPUTS:
            ref_volume (np.array): Reference volume to realign to

            realign_volume (np.array): Volume to realign on the ref_volume

            affine_transform (np.array): Affine transform with translation and scale change information 
                (doesn't support rotations)

            axis (int): slice axis to realign on length of this axis will change at the output
        
        OUTPUTS:
            ref_volume, rescaled_volume: arrays of realigned volumes with the same dimensions from the input volumes"""

    # Vertical shift in nbre of voxels (slice_delta = mm_delta /slice_height[in mm])
    shift = -int(affine_transform[2, -1] /affine_transform[2,2]) 

    # Selects the shared slices
    if shift >= 0:
        realign_volume = realign_volume[..., shift:]
        ref_volume = ref_volume[..., :realign_volume.shape[-1]]        
    else:
        ref_volume = ref_volume[..., -shift:realign_volume.shape[-1]]
        realign_volume = realign_volume[..., :ref_volume.shape[-1]]

    # Initializes the pystackreg affine transform (only slice wise wise translation)
    sr = StackReg(StackReg.AFFINE)
    aff_transform = np.array([[affine_transform[0,0], 0, affine_transform[0,0] *affine_transform[0,-1]], 
                                    [0, affine_transform[1,1],  affine_transform[1,1] *affine_transform[1,-1]], 
                                    [0, 0, 1]])
    
    slc = [slice(None)] *len(ref_volume)
    rescaled_volume = np.zeros_like(realign_volume)
    # Slice wise translation 
    for i in range(realign_volume.shape[axis]):
        slc[axis] = slice(i,i+1)
        print("\rAffine transformation on slice {}/{}".format(i+1, realign_volume.shape[axis]), end=' '*5)
        rescaled_volume[tuple(slc)] = sr.transform_stack(realign_volume[tuple(slice(None), slc)], tmats=aff_transform[None,:])

    print('')
    return ref_volume, rescaled_volume


def coregister(ref_volume, moving_volume, fill_value=-1024, resampling_only=False, plot_fig=False):
    """Coregisters moving_volume to ref_volume with scale change using SimpleITK framework using Euler3DTransform (scale change, translation rotation)
        PARAMS:
            ref_volume (np.array): Reference volume to realign to

            realign_volume (np.array): Volume to realign on the ref_volume

            fill_value (int): Fill the array where the transformation creates out of frame pixels

            resampling_only (bool): Resamples the moving and ref volumes at the same coordonates determined by their pixel spacing and origin
                if True "dumb" coregistration not optimized Euler3DTransform

            plot_fig (bool): Whether to plot the results for debugging

        RETURNS:
            ref_volume (np.array): array from ref_volume smpled at the same points as moving_volume"
            moving_coreg (np.array): array from moving_volume smpled at the same points as ref_volume"""

    def coreg_progress():
        nonlocal niter, level
        print("\rCoregistering... level: {}/3, step: {}/100".format(level, niter), end=' ')
        niter += 1

    def update_level():
        nonlocal niter, level
        niter = 0
        level += 1

    print("Resampling...")
    # Resamples moving_volume with the same pixel spacing and origin as ref_volume
    id_trans = sitk.Transform(ref_volume.GetDimension(), sitk.sitkIdentity)
    moving_resampled = sitk.Resample(moving_volume, ref_volume,
                                     id_trans, sitk.sitkLinear,
                                     fill_value, ref_volume.GetPixelID())

    # Optimized Euler3DTransform computation
    if not resampling_only:
        registration_method = sitk.ImageRegistrationMethod()
        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                          convergenceMinimumValue=1e-8, convergenceWindowSize=5)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        niter, level = 0, 1
        registration_method.AddCommand(sitk.sitkIterationEvent, coreg_progress) 
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_level) 
        
        optimized_transform = sitk.Euler3DTransform()
        registration_method.SetInitialTransform(optimized_transform, inPlace=True)

        #Computes optimal transform
        registration_method.Execute(sitk.Cast(ref_volume, sitk.sitkFloat32),
                                    sitk.Cast(moving_resampled, sitk.sitkFloat32))
        #TODO Add progress bar here
        print("Done!")

        with open('log.txt', mode='wt') as log:
            # changes print output target to log file
            defauld_stdout = sys.stdout
            sys.stdout = log
            
            # prints usefull info
            print("Optimizer's stopping condition, {0}".format(registration_method.GetOptimizerStopConditionDescription()))
            print("Final metric value: {0}".format(registration_method.GetMetricValue()), end='\n'*2)
            print(optimized_transform)
            print('\n'*2, "Transforms (0,0,0) to: {}".format(optimized_transform.TransformPoint((0.0, 0.0, 0.0))))

            sys.stdout = defauld_stdout
            
        moving_coreg = sitk.Resample(moving_resampled,
                                     optimized_transform, sitk.sitkLinear,
                                     fill_value, ref_volume.GetPixelID())
        
        # Blank space left at the top of one of the volumes
        shift = (optimized_transform.GetTranslation()[2]) /ref_volume.GetSpacing()[2]
        # Maximum displacement caused by rotation of the volume considering image dims are isotropic in the plane
        delta_z = ref_volume.GetSize()[0] *ref_volume.GetSpacing()[0] *(np.tan(optimized_transform.GetAngleX()) +np.tan(optimized_transform.GetAngleY())) /ref_volume.GetSpacing()[2]
        print(shift, delta_z)
        shift = int(round(abs(shift) +abs(delta_z)))
        
    else:
        moving_coreg = moving_resampled        

    if plot_fig:
        fig, axes = plt.subplots(1, 3)
        tifffile.imshow((sitk.GetArrayViewFromImage(ref_volume) > 0), cmap='gray',
                        figure=fig, subplot=axes[0])
        tifffile.imshow((sitk.GetArrayViewFromImage(moving_resampled) > 0), cmap='gray',
                        figure=fig, subplot=axes[1])
        tifffile.imshow(sitk.GetArrayViewFromImage(moving_coreg) > 0, cmap='gray',
                        figure=fig, subplot=axes[2])
        plt.show()
    
    # Computes valid slice to not have empty images
    valid_slices = slice(shift, moving_volume.GetDepth() -shift)
    print(valid_slices)
    return ref_volume[:,:, valid_slices], moving_coreg[:,:, valid_slices]