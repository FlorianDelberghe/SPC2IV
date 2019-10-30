from __future__ import division, print_function

import datetime
import json
import os
import shutil
import sys

import keras
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.external.tifffile as tifffile
import tensorflow as tf
from keras import backend as K
from keras.activations import sigmoid
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Dense, Dropout, Flatten, Input, MaxPool2D,
                          MaxPooling2D, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l1
from keras_contrib.layers.normalization.instancenormalization import \
    InstanceNormalization

import code.utilities as utilities
from code.data_loader import DataLoaderCT, DataLoaderDICOM

plt.rcParams['image.cmap'] = 'gray'


class CycleGAN():
    """Mother class for the different types of GANs"""

    def __init__(self, name, **kwargs):
        """Sets the basic parameters that are common to all of the GANs"""

        self.name = name
        # Default value on case they were no tset in kwargs
        defaults = {'save_state': False, 'input_res': (512,512), 'n_channels': 1, 
                    'dataset_name': 'spc2iv', 'data_process': 'coreg_sig', 'data_contrasts': ['spc', 'iv'],
                    'g_filters': 32, 'd_filters': 64, 
                    'D_loss': 'mse', 'D_loss_weight': 1, 'cycle_loss_weight': 10.0,
                    'initial_lr': 0.0002, 'start_epoch': 0}

        for key in defaults.keys():
            kwargs.setdefault(key, defaults[key])

        self.save_state = kwargs['save_state']
        self.del_out_dirs(**kwargs)

        # Input shape
        self.img_rows = kwargs['input_res'][0]
        self.img_cols = kwargs['input_res'][1]
        self.channels = kwargs['n_channels']
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = kwargs['dataset_name']
        self.data_process = kwargs['data_process']
        self.data_contrasts = kwargs['data_contrasts']

        #Loads from PNG
        if kwargs['load_from'].lower() == 'png':            
            self.data_loader = DataLoaderCT('data/images/{}'.format(self.data_process),
                                            contrasts=self.data_contrasts, image_res=(self.img_rows, self.img_cols), 
                                            n_channels=self.channels)

        # Loads from dicom and windowing
        elif kwargs['load_from'].lower() == 'dicom':
            self.data_loader = DataLoaderDICOM('data/dicom/', contrasts=self.data_contrasts,
                                               image_res=(self.img_rows, self.img_cols),
                                               channels=['soft', 'bone'])

        else:
            raise ValueError("Can't load from {!r}".format(kwargs['load_from']))

        # Sets number of filters for Generator and Discriminator
        self.gf = kwargs['g_filters']
        self.df = kwargs['d_filters']

        # Sets losses types and weights
        self.D_loss = kwargs['D_loss']
        self.lambda_cycle = kwargs['cycle_loss_weight']
        self.lambda_D = kwargs['D_loss_weight']

        self.initial_lr = kwargs['initial_lr']
        self.current_epoch = kwargs['start_epoch']


    def __call__(self, image, contrast):
        """Predicts the output of the network on an image of given contrast (forward pass)"""        

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=[0,-1])

        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        #TODO assertion error not reliable when debug=False
        assert image.shape[1:3] == self.img_shape[:2], "image shape: {} != self.image_res: {}".format(image.shape[1:3], self.img_shape[:2])
        assert image.shape[-1] == self.channels, "image channels: {} != self.channels: {}".format(image.shape[3], self.channels)

        if contrast.lower() == self.data_contrasts[0].lower():
            output = self.g_AB.predict(image)

        elif contrast.lower() == self.data_contrasts[1].lower():
            output = self.g_BA.predict(image)

        else:
            raise NotImplementedError("'{}' is not a valid contrast, ...yet".format(contrast))

        return output


    def __repr__(self):
        """Returns str with usefull info about the params of the network"""

        s = "{}({})".format(self.__class__.__name__, '{}')
        for key, value in vars(self).items():
            if isinstance(value, (float, int, str, list, dict, tuple)):
                s = s.format("{!r}={!r}, {}".format(key, value, '{}'))

        return s.format('')


    def __str__(self):
        return self.__repr__()

    def build_generator(self, out_activation='tanh', architecture='unet',
                        transformer_layers=None, normalisation='instance', **kwargs):
        """Returns a custom cycleGAN generator given custom parameters
            PARAMS:
                self (CycleGAN):
                out_activaltion (str): activation for the last layer of the generator can be any value passed to the activation kwargs of keras.layers.Dense()
                    defaults: 'tanh' can also take values 'sigmoid', None...

                architechture: (str): How the encoder/decoder architechture is built 'unet' to add skip connections between encoder and decoder 'auto-encoder' otherwise
                    default: 'unet'

                transformer_layers (int): How many transformer layers (ResNet like) to transform the low dim representation of the data to add
                    default: None

                normalization (str): How the layers inside the network will be normaized takes values 'instance' or 'batch', case insensitive
                    defaults: 'instance'            
            """

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""

            nonlocal Normalization

            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = Normalization()(d)
            return d

        def resnet_transformer(layer_input, filters, n_blocks=6):
            """Transformer network to convert from one low dimension representaion to another"""

            def res_block(layer_input, filters, f_size=4):
                """Res block used for the transformer"""

                nonlocal Normalization

                r = Conv2D(filters*2, kernel_size=f_size, strides=2, padding='same', activation='relu')(layer_input)
                r = UpSampling2D(size=2)(r)
                r = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(r)
                r = Normalization()(r)

                r = Add()([r, layer_input])
                r = Activation('relu')(r)

                return r

            assert n_blocks > 0, "'n_blocks' must be a positive integer, is: {}".format(n_blocks)

            #Builds the blocks of the residual network
            r = res_block(layer_input, filters)
            for _ in range(1, n_blocks):
                r = res_block(r, filters)

            return r        

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0, architecture='UNet'):
            """Layers used during upsampling"""
            nonlocal Normalization

            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            #TODO use LeakyReLU here ?
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = Normalization()(u)

            # Concats the output if UNet encoder/decoder
            if architecture.lower() ==  'unet':
                u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        if normalisation.lower() == 'instance':
            Normalization = InstanceNormalization

        elif normalisation.lower() == 'batch':
            Normalization = BatchNormalization

        else:
            raise ValueError("Invalid value for normatlization method: {!r}".format(normalisation))

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        if transformer_layers is not None:
            if transformer_layers > 0:
                # Transforming then Upsampling
                res = resnet_transformer(d4, self.gf*8, transformer_layers)
                u1 = deconv2d(res, d3, self.gf*4, architecture=architecture)

        else:
            # Upsampling only
            u1 = deconv2d(d4, d3, self.gf*4, architecture=architecture)

        u2 = deconv2d(u1, d2, self.gf*2, architecture=architecture)
        u3 = deconv2d(u2, d1, self.gf, architecture=architecture)
        u4 = UpSampling2D(size=2)(u3)

        # Image input
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation=out_activation)(u4)

        return Model(d0, output_img)


    def build_discriminator(self, D_output='patch', D_out_activation='sigmoid', normalisation='instance', **kwargs):
        """Builds discriminator network
            PARAMS:
                self (CycleGAN):
                D_output (str): The shape of the discriminator output 'patch' for patch GAN architecture 'single_value' adds fully connected
                    layers after the patch with one value output

                D_out_activation (str): Activation of the last layer of the discriminator any value accepted by keras.layers.Dense() activation kwarg

                normalization (str): How the layers inside the network will be normaized takes values 'instance' or 'batch', case insensitive
                    defaults: 'instance'    
                """

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator CNN layer"""

            nonlocal Normalization

            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = Normalization()(d)
            return d

        img = Input(shape=self.img_shape)

        # Sets normalization value
        if normalisation.lower() == 'instance':
            Normalization = InstanceNormalization

        elif normalisation.lower() == 'batch':
            Normalization = BatchNormalization

        else:
            raise ValueError("Invalid value for normatlization method: {!r}".format(normalization))
        
        # Convolutional layers
        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)        

        # Dsicriminator with in_res /2**4 patch output
        if D_output.lower() == 'patch':
            # Shapes of the outputs ised to build tagets
            self.D_out_shape = (*tuple(map(lambda x: int(x / (2**4)), self.img_shape[:2])), 1)

            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation=D_out_activation)(d4)   

            return Model(img, validity)

        # Discriminator with single value output 
        elif D_output.lower() == 'single_value':
            self.D_out_shape = (1)

            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='relu')(d4) 

            # Fully connected network
            pool = MaxPooling2D()(validity)
            fc1 = Flatten(data_format='channels_last')(pool)
            drop = Dropout(rate=0.3)(fc1)
            fc2 = Dense(units=16, activation='relu')(drop)
            out = Dense(units=1, activation=D_out_activation)(fc2)

            return Model(img, out)

        else: 
            raise ValueError("Invalid output type for discriminator: '{}'".format(D_output))

    
    def train(self, *args, **kwargs):
        "Virtual method to by implemented by daughters classes"

        raise NotImplementedError("Must be implemented in daughter class")


    def sample_images(self, epoch, batch_i):
        """Samples images from the validation set and predicts net output on them for visual evaluation"""

        os.makedirs("images/{}".format(self.dataset_name), exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(contrast=self.data_contrasts[0], batch_size=1)
        imgs_B = self.data_loader.load_data(contrast=self.data_contrasts[1], batch_size=1)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        # Squeeze the batch_size dim of the data
        gen_imgs = [imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B]
        gen_imgs = [np.squeeze(img, axis=0) for img in gen_imgs]

        titles = ['Original', 'Translated', 'Reconstructed']
        contrast = [cont.upper() for cont in self.data_contrasts]

        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):

                # Plot soft Tissue contrast only
                if self.channels == 1:
                    axs[i,j].imshow(np.squeeze(gen_imgs[i*c +j], axis=-1), cmap='gray')
                # Soft and bone contrast
                elif self.channels == 2:                    
                    axs[i,j].imshow(gen_imgs[i*c +j][...,0], cmap='gray')
                else:
                    axs[i,j].imshow(gen_imgs[i*c +j][...,1], cmap='gray')

                axs[i, j].set_title("{} {}".format(titles[j], contrast[(i*c+j) %2]))
                axs[i,j].axis('off')

        # Saves teh images
        if self.save_state:
            print("Saving to: /images/{}/{}_{}.png".format(self.dataset_name, epoch, batch_i))
            fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i), dpi=400)
            plt.close()

        else:
            print("Could not save model")


    def predict_volume(self, volume, contrast, axis=0):
        """Predicts the Generator output on a whole volume
            PARAMS:
                volume (np.array): Volume to transform
                contrast (str): Contrast of the image volume used to select the right generator 'spc' or 'iv' case insensitive
                axis (int): axis along which to slice the volume
            
            RETURNS: 
                trans_volume (np.array): transformed volume"""

        if contrast.lower() == self.data_contrasts[0].lower():
            generator = self.g_AB

        elif contrast.lower() == self.data_contrasts[1].lower():
            generator = self.g_BA

        else:
            raise ValueError("Invalid value for contrast: {!r}".format(contrast))

        pred_volume = np.empty_like(volume)
        slc = [slice(None)] *len(volume.shape)

        for i in range(volume.shape[axis]):

            print("\rSlice: {:3>d}".format(i), end='')
            slc[axis] = slice(i,i+1)
            pred_volume[tuple(slc)] = generator.predict(volume[tuple(slc)])

        print('')
        
        return pred_volume


    def plot_progress(self, epoch, D_loss, D_accuracy, G_loss, G_adv, G_recon):
        """Creates a figure with the values for the loss and accuracy at each epoch"""

        if epoch == 0:
            self.progress = {}
            self.progress['epochs'] = [epoch]

        else:
            self.progress['epochs'].append(epoch)

        colors = ['r', 'g', 'b', 'k', 'm']
        line_width = [2, 1, 2, 1, 1]
        legends = ['D_loss', 'D_accuracy', 'G_loss', 'G_adv', 'G_recon']

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i, (key, value) in enumerate(zip(['D_loss', 'D_accuracy', 'G_loss', 'G_adv', 'G_recon'],
                                             [D_loss, D_accuracy, G_loss, G_adv, G_recon])):

            if epoch == 0:
                self.progress[key] = [value]

            else:
                self.progress[key].append(value)

            ax.plot(self.progress['epochs'], self.progress[key], colors[i], lw=line_width[i], label=legends[i])
            ax.legend(loc='upper right', fontsize='x-large')

        plt.savefig('figures/progress.png', dpi=1000)

        
    def save(self, folderpath, save_D=True):
        """Saves weights for the CycleGAN's generators and descriptors networks
            PARAMS:
                folderpath (str): folder in which to save the models
                save_D (bool): whether or not to save the weights of the discriminator"""
        
        if self.save_state:
            try:
                os.makedirs(folderpath, exist_ok=True)

            except:
                print("Could not create dir: '{}'".format(folderpath))
                raise 

            print("Saving model to: {}".format(folderpath))

            json_dict = {}
            for key, value in vars(self).items(): 
                # Selects only json serializable objects
                if isinstance(value, (float, int, str, list, tuple, bool)):
                    json_dict[key] = value

            # Saves the models params to json to have information when loading
            with open('{}/param.json'.format(folderpath), 'w') as f:
                json.dump(json_dict, f, indent=0)

            # Saves weights
            if save_D:
                self.d_A.save_weights(folderpath+"/d_A.h5")
                self.d_B.save_weights(folderpath+"/d_B.h5")

            self.g_AB.save_weights(folderpath+"/g_AB.h5")
            self.g_BA.save_weights(folderpath+"/g_BA.h5")

        else:
            print("Could not save model")


    def load_weights(self, folderpath, load_D=True):
        """Loads model weights from saves .h5 files"""

        if load_D:
            self.d_A.load_weights(folderpath+"/d_A.h5")
            self.d_B.load_weights(folderpath+"/d_B.h5")

        self.g_AB.load_weights(folderpath+"/g_AB.h5")
        self.g_BA.load_weights(folderpath+"/g_BA.h5")

    
    @staticmethod
    def updated_lr(initial_lr, cutoff_epoch, max_epoch, how=None):
        """Returns a closure that computes the value of the learning rate depending on the epoch"""

        def const_lr(epoch):
            """Constant learning rate"""
           
            return initial_lr if epoch <= max_epoch else 0.0

        def compute_lr(epoch):
            """Linearly decreasing learning rate after cutoff epoch"""
            if epoch < cutoff_epoch:
                return initial_lr
            elif epoch > max_epoch:
                return 0.0
            else:
                return initial_lr *(1 -(epoch -cutoff_epoch) /(max_epoch -cutoff_epoch))

        if how is None:
            return const_lr

        elif how == 'linear':
            return compute_lr

        else:
            raise NotImplementedError


    @staticmethod
    def update_lr(optimizer, new_lr):
        """Updates the learning rate of the optimizer"""

        K.set_value(optimizer.lr, new_lr)
        print("Change lr for optimizer: {}".format(K.get_value(optimizer.lr))) 


    @staticmethod
    def del_out_dirs(save_state=False, out_dirs=['models', 'images'], dataset_name='spc2iv', **kwargs):
        """Deletes what was previously in the output directories if save_state is True"""

        if save_state:
            for folder in out_dirs:
                try:
                    shutil.rmtree("{}/{}/".format(folder, dataset_name))
                except FileNotFoundError:
                    print("Failed to delete folder '{}/{}'".format(folder, dataset_name))
                else:
                    print("Succesfully deleted folder '{}/{}'".format(folder, dataset_name))


    @staticmethod
    def soft_D_target(batch_size, dim, channels):
        """Softens the target of the net (Not used!)"""

        x, y = np.meshgrid(np.linspace(-1,1, dim[0]), np.linspace(-1,1, dim[1]))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 1.2, 0.0
        g = np.exp(-((d-mu)**2 / (2.0 *sigma**2)))

        g = 0.4*np.stack([g]*channels, axis=-1)

        return np.stack([g]*batch_size, axis=0)

#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

class BasicCycleGAN(CycleGAN):

    def __init__(self, **kwargs):

        super(BasicCycleGAN, self).__init__('BasicCycleGAN', **kwargs)

        # Network Losses, d_loss set in super
        self.cycle_loss = 'mae'
        self.id_loss = 'mae'

        # Loss weights
        self.lambda_id = kwargs['id_loss_weight'] if 'id_loss_weight' in kwargs.keys() else 1.0

        self.lr = self.updated_lr(self.initial_lr, kwargs['cutoff_epoch'], 
                                  kwargs['max_epochs'], kwargs['optimizer_scheduler'])
        optimizer = Adam(self.lr(0), 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(**kwargs)
        self.d_B = self.build_discriminator(**kwargs)

        self.d_A.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Build the generators
        self.g_AB = self.build_generator(**kwargs)
        self.g_BA = self.build_generator(**kwargs)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])

        self.combined.compile(loss=[self.D_loss, self.D_loss,
                                    self.cycle_loss, self.cycle_loss,
                                    self.id_loss, self.id_loss],
                              loss_weights=[self.lambda_D, self.lambda_D,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)


    def train(self, epochs, batch_size=1, sample_interval=50, starting_epoch=0):
        
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size, *self.D_out_shape))
        fake = np.zeros((batch_size, *self.D_out_shape))

        for epoch in range(starting_epoch, epochs):
            self.current_epoch = epoch
            self.update_lr(self.combined.optimizer, self.lr(epoch)) 
            
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, paired=False)):       

                # Train Discriminators
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Prints the progress
                print ("[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [D loss:{:8.5f}, acc:{:3.0f}%] [G loss: {:8.5f}, adv: {:8.5f}, recon: {:8.5f}, id: {:8.5f}] time: {:s}".format(
                    epoch+1, epochs, batch_i+1, self.data_loader.n_batches //self.data_loader.one_in_x, d_loss[0], 100*d_loss[1], 
                    g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), str(elapsed_time)))

                # If at save interval => save generated image samples
                if (batch_i *batch_size) %sample_interval < batch_size:
                    self.sample_images(epoch+1, batch_i *batch_size)

            else:
                #Saves the net every other epoch
                if epoch %5 == 0 and epoch > 0:
                    self.save("models/{}/{}_epoch_{}".format(self.dataset_name, self.name, epoch))
                #TODO rolling averages of the losses to be more accurate
                self.plot_progress(epoch, d_loss[0], d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]))
                self.data_loader.update_val()
        else:
            # Saves final model
            self.save("models/{}/{}".format(self.dataset_name, "{}_final".format(self.name)))

            # Predicts and saves the test set
            utilities.save_volume(self.predict_volume(self.data_loader.load_test('IV'), 'IV'),
                "models/{}".format(self.dataset_name), 'final_spc_pred.tif', bit_depth=32)
            utilities.save_volume(self.predict_volume(self.data_loader.load_test('SPC'), 'SPC'),
                "models/{}".format(self.dataset_name), 'final_iv_pred.tif', bit_depth=32)

#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

class PairedLossCycleGAN(CycleGAN):

    def __init__(self, **kwargs):

        super(PairedLossCycleGAN, self).__init__('PairedLossCycleGAN', **kwargs)

        # Network Losses, d_loss set in super
        self.pair_loss = 'mae'
        self.cycle_loss = 'mae'
        # Loss weights, lambda_cycle and lambda_D set in super
        self.lambda_pair = kwargs['pair_loss_weight'] if 'pair_loss_weight' in kwargs.keys() else 6.0

        self.lr = self.updated_lr(self.initial_lr, kwargs['cutoff_epoch'], 
                                  kwargs['max_epochs'], kwargs['optimizer_scheduler'])
        
        optimizer = Adam(self.initial_lr, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(**kwargs)
        self.d_B = self.build_discriminator(**kwargs)
        
        self.d_A.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        self.d_B.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Construct Computational Graph of Generators 
        # Build the generators
        self.g_AB = self.build_generator(**kwargs)
        self.g_BA = self.build_generator(**kwargs)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)        

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False
        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       fake_B, fake_A,
                                       reconstr_A, reconstr_B])

        self.combined.compile(loss=[self.D_loss, self.D_loss,
                                    self.pair_loss, self.pair_loss,
                                    self.cycle_loss, self.cycle_loss],
                              loss_weights=[self.lambda_D, self.lambda_D,
                                            self.lambda_pair, self.lambda_pair,
                                            self.lambda_cycle, self.lambda_cycle],
                              optimizer=optimizer)


    def train(self, epochs, batch_size=1, sample_interval=50, starting_epoch=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size, *self.D_out_shape))
        fake = np.zeros((batch_size, *self.D_out_shape))

        for epoch in range(starting_epoch, epochs+1):
            self.current_epoch = epoch
            self.update_lr(self.combined.optimizer, self.lr(epoch)) 

            for batch_i, (imgs_A, imgs_B, paired_A, paired_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       paired_A, paired_B,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Prints the progress
                print("[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [D loss:{:8.5f}, acc:{:3.0f}] [G loss: {:8.5f}, adv: {:8.5f}, recon: {:8.5f}] time: {:s}".format(
                    epoch+1, epochs, batch_i+1, self.data_loader.n_batches // self.data_loader.one_in_x, d_loss[0], 100*d_loss[1],
                    g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), str(elapsed_time)))

                # If at save interval => save generated image samples
                if (batch_i *batch_size) %sample_interval < batch_size:
                    self.sample_images(epoch+1, batch_i *batch_size)

            else:
                #Saves the net every other epoch
                if epoch %5 == 0 and epoch > 0:
                    self.save("models/{}/{}_epoch_{}".format(self.dataset_name, self.name, epoch))
                #TODO rolling averages of the losses to be more accurate
                self.plot_progress(epoch+1, d_loss[0], d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]))
                self.data_loader.update_val()
        else:
            # Saves final model
            self.save("models/{}/{}".format(self.dataset_name, "{}_final".format(self.name)))

            # Predicts and saves the test set
            utilities.save_volume(self.predict_volume(self.data_loader.load_test('IV'), 'IV'),
                "models/{}".format(self.dataset_name), 'final_spc_pred.tif', bit_depth=1326)
            utilities.save_volume(self.predict_volume(self.data_loader.load_test('SPC'), 'SPC'),
                "models/{}".format(self.dataset_name), 'final_iv_pred.tif', bit_depth=32)

#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

class CMapCycleGAN(CycleGAN):

    def __init__(self, **kwargs):
        
        super(CMapCycleGAN, self).__init__('CMapCycleGAN', **kwargs)

        # Loss weights
        self.lambda_pair = 0.8 *self.lambda_cycle
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_A.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        self.d_B = self.build_discriminator()
        self.d_B.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Construct Computational Graph of Generators 
        # Build the generators
        self.g_AB = self.build_generator(**kwargs)
        self.g_BA = self.build_generator(**kwargs)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        cmap_A = Input(shape=self.img_shape)
        cmap_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_contrast_B = self.g_AB([img_A, cmap_A])
        fake_contrast_A = self.g_BA([img_B, cmap_B])

        # Translate images back to original domain
        # img_A +fake_cmap_B ~= fake_B
        reconstr_contrast_A = self.g_BA([img_A, fake_contrast_B])
        reconstr_contrast_B = self.g_AB([img_B, fake_contrast_A])

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A([img_B, fake_contrast_A])
        valid_B = self.d_B([img_A, fake_contrast_B])

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B, cmap_A, cmap_B],
                              outputs=[valid_A, valid_B,
                                       fake_contrast_B, fake_contrast_A,
                                       reconstr_contrast_A, reconstr_contrast_B])

        self.combined.compile(loss=[self.D_loss, self.D_loss,
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[self.lambda_D, self.lambda_D,
                                            self.lambda_pair, self.lambda_pair,
                                            self.lambda_cycle, self.lambda_cycle],
                              optimizer=optimizer)


    def build_generator(self, **kwargs):
        """UNet like Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        # def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        #     """Layers used during upsampling"""
        #     u = UpSampling2D(size=2)(layer_input)
        #     u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        #     if dropout_rate:
        #         u = Dropout(dropout_rate)(u)
        #     u = InstanceNormalization()(u)
        #     u = Concatenate()([u, skip_input])
        #     return u

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            return u

        # Image input
        d_in = Input(shape=self.img_shape)
        cmap = Input(shape=self.img_shape)

        d0 = Add()([d_in, cmap])

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        # u1 = deconv2d(d4, d3, self.gf*4)
        # u2 = deconv2d(u1, d2, self.gf*2)
        # u3 = deconv2d(u2, d1, self.gf)
        # u4 = UpSampling2D(size=2)(u3)

        u1 = deconv2d(d4, self.gf*4)
        u2 = deconv2d(u1, self.gf*2)
        u3 = deconv2d(u2, self.gf)
        u4 = UpSampling2D(size=2)(u3)

        output_contrast_map = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(inputs=[d_in, cmap], outputs=output_contrast_map)


    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)
        cmap = Input(shape=self.img_shape)

        d0 = Add()([img, cmap])

        d1 = d_layer(d0, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        if self.D_loss == 'mse':
            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(d4)
        elif self.D_loss == 'binary_crossentropy':
            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)
        else:
            raise ValueError("'{}' is not a valid Dicriminator Loss".format(self.D_loss))        

        # pool = MaxPooling2D()(validity)
        # fc1 = Flatten(data_format='channels_last')(pool)
        # drop = Dropout(rate=0.3)(fc1)
        # fc2 = Dense(units=16, activation='relu')(drop)
        # out = Dense(units=1, activation='sigmoid')(fc2)

        return Model(inputs=[img, cmap], outputs=validity)


    def pre_train_generators(self, epochs=1, batch_size=1, sample_interval=200):

            valid = np.ones((batch_size, 32, 32, 1))
            fake = np.zeros((batch_size, 32, 32, 1))

            for epoch in range(1, epochs+1):
                for batch_i, (imgs_A, imgs_B, paired_A, paired_B) in enumerate(self.data_loader.load_batch(batch_size)):
                    print ("[Epoch {:3d}/{:d}] [Batch {:4d}/{:d}]".format(epoch, epochs, batch_i, 
                            self.data_loader.n_batches //self.data_loader.one_in_x), end=' ')

                    contrast_AB = paired_A -imgs_A
                    contrast_BA = paired_B -imgs_B
                    contrast_null = np.zeros((batch_size, *self.img_shape))

                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B, contrast_null, contrast_null],
                                                          [valid, valid,
                                                           contrast_AB, contrast_BA,
                                                           -contrast_AB, -contrast_BA])

                    # Plot the progress
                    print("[G loss: {:8.5f}]".format(g_loss[0]))

                    if int(batch_i *batch_size) %sample_interval == 0:
                        self.sample_images(epoch, batch_i)     


    def train(self, epochs, batch_size=1, sample_interval=50, starting_epoch=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        valid = np.ones((batch_size, 32, 32, 1))
        fake = np.zeros((batch_size, 32, 32, 1))

        for epoch in range(starting_epoch, epochs+1):

            # if epoch == 1:
            #     self.pre_train_generators()

            for batch_i, (imgs_A, imgs_B, paired_A, paired_B) in enumerate(self.data_loader.load_batch(batch_size)):
                print("[Epoch {:3d}/{:d}] [Batch {:4d}/{:d}]".format(
                    epoch, epochs, batch_i, self.data_loader.n_batches // self.data_loader.one_in_x), end=' ')
                
                contrast_AB = paired_A -imgs_A
                contrast_BA = paired_B -imgs_B
                contrast_null = np.zeros((batch_size, *self.img_shape))

                # Train Discriminators
                fake_cont_B = self.g_AB.predict([imgs_A, contrast_null])
                fake_cont_A = self.g_BA.predict([imgs_B, contrast_null])

                dA_loss_real = self.d_A.train_on_batch([imgs_A, contrast_null], valid)
                dA_loss_fake = self.d_A.train_on_batch([imgs_B, fake_cont_A], fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch([imgs_B, contrast_null], valid)
                dB_loss_fake = self.d_B.train_on_batch([imgs_A, fake_cont_B], fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                print ("[D loss: {:8.5f}, acc: {:3.0f}%]".format(d_loss[0], 100*d_loss[1]), end=' ')
                
                # Train Generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, contrast_null, contrast_null],
                                                        [valid, valid,
                                                        contrast_AB, contrast_BA,
                                                        -contrast_AB, -contrast_BA])

                elapsed_time = datetime.datetime.now() -start_time
                print(" [G loss: {:8.5f}, adv: {:8.5f}, recon: {:8.5f}] time: {:s} ".format \
                        (g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]),
                        str(elapsed_time)))

                # Save generated image samples at sample_interval
                if int(batch_i *batch_size) %sample_interval == 0:
                    self.sample_images(epoch, batch_i)

            else:
                #Saves the net every other epoch
                if epoch %5 == 0:
                    self.save("models/{}/epoch_{}".format(self.dataset_name, epoch))
                self.sample_images(epoch, self.data_loader.n_batches //self.data_loader.one_in_x)
                self.data_loader.update_val()
        else:
            # Saves final model
            self.save("models/{}/{}".format(self.dataset_name, 'final'))
            #TODO predict the whole test set


    def sample_images(self, epoch, batch_i):
        """Samples images from the validation set and predicts net output on them for visual evaluation"""

        os.makedirs("images/{}".format(self.dataset_name), exist_ok=True)
        r, c = 2, 3

        imgs_SPC = self.data_loader.load_data(contrast='spc', batch_size=1)
        imgs_IV = self.data_loader.load_data(contrast='iv', batch_size=1)

        contrast_null = np.zeros((1, *self.img_shape))
        # Translate images to the other domain
        fake_cont_IV = self.g_AB.predict([imgs_SPC, contrast_null])
        fake_cont_SPC = self.g_BA.predict([imgs_IV, contrast_null])
        # Translate back to original domain
        reconstr_cont_SPC = self.g_BA.predict([imgs_SPC, fake_cont_IV])
        reconstr_cont_IV = self.g_AB.predict([imgs_IV, fake_cont_SPC])
        # Squeeze the batch_size dim of the data
        gen_imgs = [imgs_SPC, imgs_SPC +fake_cont_IV, imgs_SPC +fake_cont_IV +reconstr_cont_SPC,
                    imgs_IV, imgs_IV +fake_cont_SPC, imgs_IV +fake_cont_SPC +reconstr_cont_IV]
        gen_imgs = [np.squeeze(img, axis=0) for img in gen_imgs]

        titles = ['Original', 'Translated', 'Reconstructed']
        contrast = ['SPC', 'IV']
        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                if self.channels == 1:
                    axs[i,j].imshow(np.squeeze(gen_imgs[i*c +j], axis=-1), cmap='gray')
                else:
                    axs[i,j].imshow(gen_imgs[i*c +j][...,1], cmap='gray')
                axs[i, j].set_title("{} {}".format(titles[j], contrast[(i*c+j) %2]))
                axs[i,j].axis('off')

        if self.save_state:
            print("Saving to: /images/{}/{}_{}.png".format(self.dataset_name, epoch, batch_i))
            fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i), dpi=400)
            plt.close()
        else:
            print("Could not save model")

#------------------------------------------------------------#
#
#
#------------------------------------------------------------#

class ResCycleGAN(CycleGAN):

    def __init__(self, **kwargs):
        super(ResCycleGAN, self).__init__('ResCycleGAN', **kwargs)

        # Network Losses, d_loss set in super
        self.id_loss = 'mae'
        self.cycle_loss = 'mae'
        # Loss weights, lambda_cycle and lambda_D set in super
        self.lambda_id = kwargs['id_loss_weight'] if 'id_loss_weight' in kwargs.keys() else 1.0

        self.lr = self.updated_lr(self.initial_lr, kwargs['cutoff_epoch'], 
                                  kwargs['max_epochs'], kwargs['optimizer_scheduler'])
        
        optimizer = Adam(self.lr(0), 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(**kwargs)
        self.d_B = self.build_discriminator(**kwargs)
        
        self.d_A.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        self.d_B.compile(loss=self.D_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Construct Computational Graph of Generators 
        # Build the generators
        self.g_AB = self.build_generator(**kwargs)
        self.g_BA = self.build_generator(**kwargs)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)        

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])

        self.combined.compile(loss=[self.D_loss, self.D_loss,
                                    self.cycle_loss, self.cycle_loss,
                                    self.id_loss, self.id_loss],
                              loss_weights=[self.lambda_D, self.lambda_D,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)


    def build_generator(self, out_activation='sigmoid', architecture='UNet', transformer_layers=None, **kwargs):
        """Returns a custom cycleGAN generator given custom parameters"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            nonlocal Normalization
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = Normalization()(d)
            return d

        def resnet_transformer(layer_input, filters, n_blocks=6):

            def res_block(layer_input, filters, f_size=4):
                nonlocal Normalization
                r = Conv2D(filters*2, kernel_size=f_size, strides=2, padding='same', activation='relu')(layer_input)

                r = UpSampling2D(size=2)(r)
                r = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(r)

                r = Normalization()(r)

                r = Add()([r, layer_input])
                r = Activation('relu')(r)
                return r

            assert n_blocks > 0, "'n_blocks' must be a positive integer, is: {}".format(n_blocks)

            r = res_block(layer_input, filters)
            for _ in range(1, n_blocks):
                r = res_block(r, filters)

            return r        

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0, architecture='UNet'):
            """Layers used during upsampling"""
            nonlocal Normalization
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = Normalization()(u)

            # Concats the output if UNet encoder/decoder
            if architecture.lower() == 'unet':
                u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Set the normalization method depending on batch size
        if self.img_shape[0] == 1:
            Normalization = InstanceNormalization
        else:
            Normalization = BatchNormalization

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        if transformer_layers is not None:
            # Transforming then Upsampling
            res = resnet_transformer(d4, self.gf*8, transformer_layers)
            u1 = deconv2d(res, d3, self.gf*4, architecture=architecture)
        else:
            # Upsampling only
            u1 = deconv2d(d4, d3, self.gf*4, architecture=architecture)

        u2 = deconv2d(u1, d2, self.gf*2, architecture=architecture)
        u3 = deconv2d(u2, d1, self.gf, architecture=architecture)
        u4 = UpSampling2D(size=2)(u3)

        # Contrast map predicted
        res_image = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation=None, 
                           activity_regularizer=l1(1e-4))(u4)
        res_image = LeakyReLU(alpha=0.2)(res_image)
        # res_image = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        
        # add_image = Add()([d0, res_image])
        # output_img = Conv2D(self.channels, kernel_size=2, strides=1, padding='same', activation=out_activation)(add_image)

        add_image = Concatenate()([d0, res_image])
        output_img = Conv2D(self.channels, kernel_size=1, strides=1, padding='same', activation=out_activation)(add_image)

        return Model(d0, output_img)


    def train(self, epochs, batch_size=1, sample_interval=50, starting_epoch=1):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size, *self.D_out_shape))
        fake = np.zeros((batch_size, *self.D_out_shape))

        for epoch in range(starting_epoch, epochs+1):
            self.current_epoch = epoch
            self.update_lr(self.combined.optimizer, self.lr(epoch)) 

            for batch_i, (imgs_A, imgs_B, *_) in enumerate(self.data_loader.load_batch(batch_size)):

                print("[Epoch {:3d}/{:d}] [Batch {:4d}/{:d}]".format(
                    epoch, epochs, batch_i, self.data_loader.n_batches // self.data_loader.one_in_x), end=' ')

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                print ("[D loss: {:8.5f}, acc: {:3.0f}%]".format(d_loss[0], 100*d_loss[1]), end=' ')

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_A,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time
                print(" [G loss: {:8.5f}, adv: {:8.5f}, recon: {:8.5f}, id: {:8.5f}] time: {:s} ".format(
                    g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]), np.mean(g_loss[5:6]),
                    str(elapsed_time)))

                

                # If at save interval => save generated image samples
                if int(batch_i *batch_size) %sample_interval == sample_interval %batch_size:
                    self.sample_images(epoch, batch_i)

            else:
                #Saves the net every other epoch
                if epoch %5 == 0:
                    self.save("models/{}/epoch_{}".format(self.dataset_name, epoch))
                self.sample_images(epoch, self.data_loader.n_batches //self.data_loader.one_in_x)
                self.data_loader.update_val()

            self.plot_progress(epoch, d_loss[0], d_loss[1], g_loss[0], np.mean(g_loss[1:3]), np.mean(g_loss[3:5]))
        else:
            # Saves final model
            self.save("models/{}/{}".format(self.dataset_name, 'final'))
            self.sample_images(epochs+1, self.data_loader.n_batches)
