import cv2
import os
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Lambda, Dense, Reshape, Input
from tensorflow.keras import Model
import imageio
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
source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Reference: https://arxiv.org/abs/1312.6114

use the fit function to train the model.
'''

__all__ = ['VAE']

class VAE:

    def __init__(self,
                enc_units = [
                    256,
                    128],
                dec_units = [
                    128,
                    256],
                interm_dim = 256,
                latent_dim = 32,
                activation = 'relu',
                kernel_initializer = 'glorot_uniform',
                kernel_regularizer = None):


        self.model = None
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

        train_data = train_data.reshape(
            (-1, self.image_size[0] * self.image_size[1] * self.image_size[2])) / 255
        train_ds = tf.data.Dataset.from_tensor_slices(
            train_data).shuffle(10000).batch(batch_size)

        test_data = test_data.reshape(
            (-1, self.image_size[0] * self.image_size[1] * self.image_size[2])) / 255
        test_ds = tf.data.Dataset.from_tensor_slices(
            test_data).shuffle(10000).batch(batch_size)

        return train_ds, test_ds

    def get_sample(self, data=None, n_samples=1, save_dir=None):

        assert data is not None, "Data not provided"

        sample_images = []
        data = data.unbatch()
        for img in data.take(n_samples):

            img = img.numpy()
            img = img.reshape(
                (self.image_size[0],
                 self.image_size[1],
                 self.image_size[2]))
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

    def sampling(self, distribution):

        z_mean = distribution[0]
        z_var = distribution[1]

        batch = tf.keras.backend.shape(z_mean)[0]
        dim = tf.keras.backend.int_shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal((batch, dim))

        return z_mean + tf.keras.backend.exp(0.5 * z_var) * epsilon

    '''
    encoder and decoder layers for custom dataset can be reimplemented by inherting this class(vae)
    '''

    def vae(self, config):

        enc_units = config['enc_units']
        encoder_layers = len(enc_units)
        dec_units = config['dec_units']
        decoder_layers = len(dec_units)
        interm_dim = config['interm_dim']
        latent_dim = config['latent_dim']
        activation = config['activation']
        kernel_initializer = config['kernel_initializer']
        kernel_regularizer = config['kernel_regularizer']

        org_inputs = Input(
            shape=self.image_size[0] *
            self.image_size[1] *
            self.image_size[2])
        x = Dense(
            enc_units[0] * 2,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(org_inputs)

        for i in range(encoder_layers):
            x = Dense(
                enc_units[i],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(x)

        x = Dense(
            interm_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(x)

        z_mean = Dense(latent_dim)(x)
        z_var = Dense(latent_dim)(x)

        # Sampling from intermediate dimensiont to get a probability density
        z = Lambda(self.sampling, output_shape=(latent_dim, ))([z_mean, z_var])

        # Encoder model
        enc_model = Model(org_inputs, [z_mean, z_var])

        latent_inputs = Input(shape=(latent_dim, ))
        outputs = Dense(
            dec_units[0] // 2,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(latent_inputs)

        for i in range(decoder_layers):
            outputs = Dense(
                dec_units[i],
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(outputs)

        final_outputs = Dense(
            self.image_size[0] *
            self.image_size[1] *
            self.image_size[2],
            activation='sigmoid')(outputs)

        # Decoder model
        dec_model = Model(latent_inputs, final_outputs)

        out = dec_model(z)
        model = Model(org_inputs, out)

        kl_loss = - 0.5 * \
            tf.math.reduce_mean(z_var - tf.math.square(z_mean) - tf.math.exp(z_var) + 1)
        model.add_loss(kl_loss)

        return model

    '''
    call build_model to intialize the layers before you train the model
    '''

    def __load_model(self):

        self.model = self.vae(self.config)

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
                    data_recon = self.model(data)
                    loss = mse_loss(data, data_recon)

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
                    save_model + '/variational_autoencoder_checkpoint')
            else:
                self.model.save_weights(
                    save_model + 'variational_autoencoder_checkpoint')

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

        generated_samples = generated_samples.reshape(
            (-1, self.image_size[0], self.image_size[1], self.image_size[2]))
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
