import sys
sys.path.append('..')

import datetime
import tensorflow as tf
from losses.mse_loss import mse_loss
from datasets.load_custom_data import load_custom_data
from datasets.load_mnist import load_mnist
from datasets.load_cifar10 import load_cifar10
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Lambda, Dense, Reshape, Input
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Reference: https://arxiv.org/abs/1312.6114

use the fit function to train the model.
'''


class VAE():

    def __init__(self):

        super(VAE, self).__init__()
        self.model = None
        self.image_size = None

    def load_data(self, data_dir=None, use_mnist=False,
                  use_cifar10=False, batch_size=32, img_shape=(64, 64)):
        '''
        choose the dataset, if None is provided returns an assertion error -> ../datasets/load_custom_data
        returns a tensorflow dataset loader
        '''
        if(use_mnist):

            train_data = load_mnist()

        elif(use_cifar10):

            train_data = load_cifar10()

        else:

            train_data = load_custom_data(data_dir)

        self.image_size = train_data.shape[1:]

        train_data = train_data.reshape(
            (-1, self.image_size[0] * self.image_size[1] * self.image_size[2])) / 255
        train_ds = tf.data.Dataset.from_tensor_slices(
            train_data).shuffle(10000).batch(batch_size)

        return train_ds

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

    def vae(self, params):

        enc_units = params['enc_units'] if 'enc_units' in params else [
            256, 128]
        encoder_layers = params['encoder_layers'] if 'encoder_layers' in params else 2
        dec_units = params['dec_units'] if 'dec_units' in params else [
            128, 256]
        decoder_layers = params['decoder_layers'] if 'decoder_layers' in params else 2
        interm_dim = params['interm_dim'] if 'interm_dim' in params else 64
        latent_dim = params['latent_dim'] if 'latent_dim' in params else 32
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None

        assert len(
            enc_units) == encoder_layers, "Dimension mismatch: length of enocoder units should match number of encoder layers"
        assert len(
            dec_units) == decoder_layers, "Dimension mismatch: length of decoder units should match number of decoder layers"

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

    def build_model(
        self,
        params={
            'encoder_layers': 2,
            'decoder_layers': 2,
            'enc_units': [
            256,
            128],
            'dec_units': [
                128,
                256],
            'interm_dim': 256,
            'latent_dim': 32,
            'activation': 'relu',
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None}):

        self.model = self.vae(params)

    def fit(self, train_ds=None, epochs=100, optimizer='Adam', print_steps=100,
            learning_rate=0.001, tensorboard=False, save_model=None):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'

        kwargs = {}
        kwargs['learning_rate'] = learning_rate
        optimizer = getattr(tf.keras.optimizers, optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        for epoch in range(epochs):

            for data in train_ds:

                with tf.GradientTape() as tape:
                    data_recon = self.model(data)
                    loss = mse_loss(data, data_recon)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

                if(steps % print_steps == 0):
                    print(
                        "Step:",
                        steps + 1,
                        'reconstruction loss:',
                        loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy(), step=steps)

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

        generated_samples = []
        for data in test_ds:
            gen_sample = self.model(data, training=False)
            gen_sample = gen_sample.numpy()
            generated_samples.append(gen_sample)

        generated_samples = np.array(generated_samples)
        generated_samples = generated_samples.reshape((-1, self.image_size[0], self.image_size[1], self.image_size[2]))
        if(save_dir is None):
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            cv2.imwrite(
                os.path.join(
                    save_dir,
                    'sample_' +
                    str(i) +
                    '.jpg'),
                sample)
