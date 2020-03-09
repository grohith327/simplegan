from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import Dense, Reshape, Flatten, MaxPool2D
import cv2
import imageio
import os
from tensorflow.keras import Model
import numpy as np
from ..datasets.load_cifar10 import load_cifar10_AE
from ..datasets.load_mnist import load_mnist_AE
from ..datasets.load_custom_data import load_custom_data_AE
from ..losses.mse_loss import mse_loss
import tensorflow as tf
import datetime
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning


'''
vanilla_autoencoder imports from tensorflow Model class

Create an instance of the class and compile it by using the loss from ../losses/mse_loss and use an optimizer and metric of your choice

use the fit function to train the model.
'''

__all__ = ['ConvolutionalAutoencoder']

class ConvolutionalAutoencoder:

    def __init__(self,
                interm_dim = 128,
                enc_channels = [
                    32,
                    64],
                dec_channels = [
                    64,
                    32],
                kernel_size = (
                    5,
                    5),
                activation = 'relu',
                kernel_initializer = 'glorot_uniform',
                kernel_regularizer = None):

        self.model = tf.keras.Sequential()
        self.image_size = None
        self.config = locals()

    def load_data(self, 
                data_dir=None, 
                use_mnist=False,
                use_cifar10=False, 
                batch_size=32, 
                img_shape=(64, 64)):
        '''
        choose the dataset, if None is provided returns an assertion error -> ../datasets/load_custom_data
        returns a tensorflow dataset loader
        '''
        if(use_mnist):

            train_data, test_data = load_mnist_AE()

        elif(use_cifar10):

            train_data, test_data = load_cifar10_AE()

        else:

            train_data, test_data = load_custom_data_AE(data_dir, img_shape)

        self.image_size = train_data.shape[1:]

        train_data = train_data / 255
        train_ds = tf.data.Dataset.from_tensor_slices(
            train_data).shuffle(10000).batch(batch_size)

        test_data = test_data / 255
        test_ds = tf.data.Dataset.from_tensor_slices(
            test_data).shuffle(10000).batch(batch_size)

        return train_ds, test_ds

    def get_sample(self, data=None, n_samples=1, save_dir=None):

        assert data is not None, "Data not provided"

        sample_images = []
        data = data.unbatch()
        for img in data.take(n_samples):

            img = img.numpy()
            sample_images.append(img)

        sample_images = np.array(sample_images)

        if(save_dir is None):
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):
            imageio.imwrite(
                os.path.join(
                    save_dir,
                    'sample_' +
                    str(i) +
                    '.jpg'),
                sample)

    '''
    encoder and decoder layers for custom dataset can be reimplemented by inherting this class(vanilla_autoencoder)
    '''

    def encoder(self, config):

        enc_channels = config['enc_channels']
        encoder_layers = len(enc_channels)
        interm_dim = config['interm_dim']
        activation = config['activation']
        kernel_initializer = config['kernel_initializer']
        kernel_regularizer = config['kernel_regularizer']
        kernel_size = config['kernel_size']

        model = tf.keras.Sequential()

        model.add(
            Conv2D(
                enc_channels[0] // 2,
                kernel_size=kernel_size,
                padding='same',
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_shape=self.image_size))
        model.add(MaxPool2D())

        for i in range(encoder_layers):
            model.add(
                Conv2D(
                    enc_channels[i],
                    kernel_size=kernel_size,
                    padding='same',
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            model.add(MaxPool2D())

        model.add(Flatten())
        model.add(Dense(interm_dim, activation='sigmoid'))

        return model

    def decoder(self, config):

        dec_channels = config['dec_channels']
        decoder_layers = len(dec_channels)
        interm_dim = config['interm_dim']
        activation = config['activation']
        kernel_initializer = config['kernel_initializer']
        kernel_regularizer = config['kernel_regularizer']
        kernel_size = config['kernel_size']

        model = tf.keras.Sequential()

        model.add(
            Dense(
                (self.image_size[0] // 4) * (
                    self.image_size[1] // 4) * (
                    dec_channels[0] * 2),
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=interm_dim))

        model.add(
            Reshape(
                ((self.image_size[0] // 4),
                 (self.image_size[1] // 4),
                    (dec_channels[0] * 2))))

        k = 0
        for _ in range(decoder_layers // 2):
            model.add(
                Conv2DTranspose(
                    dec_channels[k],
                    kernel_size=kernel_size,
                    strides=(
                        1,
                        1),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            k += 1

        model.add(
            Conv2DTranspose(
                dec_channels[k],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))

        for _ in range(decoder_layers // 2):
            model.add(
                Conv2DTranspose(
                    dec_channels[k],
                    kernel_size=kernel_size,
                    strides=(
                        1,
                        1),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            k += 1

        model.add(
            Conv2DTranspose(
                self.image_size[2],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                padding='same',
                activation='tanh'))

        return model

    '''
    call build_model to intialize the layers before you train the model
    '''

    def __load_model(self):

        self.model.add(self.encoder(self.config))
        self.model.add(self.decoder(self.config))

    def fit(self, 
            train_ds=None, 
            epochs=100, 
            optimizer='Adam', 
            verbose=1,
            learning_rate=0.001, 
            tensorboard=False, 
            save_model=None):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'

        self.__load_model()

        kwargs = {}
        kwargs['learning_rate'] = learning_rate
        optimizer = getattr(tf.keras.optimizers, optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0
        train_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(train_ds).numpy()
        except:
            total = 0

        for epoch in range(epochs):

            train_loss.reset_states()

            pbar = tqdm(total = total, desc = 'Epoch - '+str(epoch+1))
            for data in train_ds:

                with tf.GradientTape() as tape:
                    recon_data = self.model(data)
                    loss = mse_loss(data, recon_data)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

                
                train_loss(loss)

                steps += 1
                pbar.update(1)

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy(), step=steps)

            pbar.close()
            del pbar

            if(verbose == 1):
                print("Epoch:",
                    epoch + 1,
                    'reconstruction loss:',
                    train_loss.result().numpy())

        if(save_model is not None):

            assert isinstance(save_model, str), "Not a valid directory"
            if(save_model[-1] != '/'):
                self.model.save_weights(
                    save_model + '/convolutional_autoencoder_autoencoder_checkpoint')
            else:
                self.model.save_weights(
                    save_model + 'convolutional_autoencoder_autoencoder_checkpoint')

    def generate_samples(self, test_ds=None, save_dir=None):

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = np.array([])
        for i, data in enumerate(test_ds):
            gen_sample = self.model(data, training=False)
            gen_sample = gen_sample.numpy()
            if(i == 0):
                generated_samples = gen_sample
            else:
                generated_samples = np.concatenate((generated_samples, gen_sample), 0)

        if(save_dir is None):
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            imageio.imwrite(
                os.path.join(
                    save_dir,
                    'sample_' +
                    str(i) +
                    '.jpg'),
                sample)
