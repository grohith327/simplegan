import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from ..datasets.load_mnist import load_mnist
from ..datasets.load_cifar10 import load_cifar10
from ..datasets.load_custom_data import load_custom_data
from ..losses.minmax_loss import gan_discriminator_loss, gan_generator_loss
from ..losses.infogan_loss import auxillary_loss
import datetime
from tqdm import tqdm
import logging
import imageio
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

__all__ = ['InfoGAN']


class InfoGAN:
    def __init__(self,
                noise_dim = 100,
                code_dim = 2,
                n_classes = 10,
                dropout_rate = 0.4,
                activation = 'leaky_relu',
                kernel_initializer = 'glorot_uniform',
                kernel_size = (
                    5,
                    5),
                gen_channels = [
                    128,
                    64
                ],
                disc_channels = [
                    64,
                    128],
                kernel_regularizer = None):
        
        
        self.image_size = None
        self.config=locals()
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.code_dim = code_dim
        
    def load_data(self,
                  data_dir=None,
                  use_mnist=False,
                  use_cifar10=False,
                  use_lsun=False,
                  batch_size=32, 
                  img_shape=(64, 64)):
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

        train_data = (train_data - 127.5) / 127.5
        train_ds = tf.data.Dataset.from_tensor_slices(
            train_data).shuffle(10000).batch(batch_size)

        return train_ds

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

    def conv_block(
            self,
            inputs,
            filters,
            kernel_size,
            strides=(
                2,
                2),
            kernel_initializer='glorot_uniform',
            kernel_regularizer=None,
            padding='same',
            activation="leaky_relu",
            use_batch_norm=True,
            conv_type="normal"):

        if(conv_type == 'transpose'):
            x = layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(inputs)
        else:
            x = layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(inputs)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        if (activation == "leaky_relu"):
            x = layers.LeakyReLU()(x)
        elif (activation == "tanh"):
            x = tf.keras.activations.tanh(x)
        return x

    def discriminator(self, config):

        dropout_rate = config['dropout_rate'] 
        disc_channels = config['disc_channels'] 
        activation = self.config['activation'] 
        kernel_initializer = config['kernel_initializer'] 
        kernel_regularizer = config['kernel_regularizer']
        kernel_size = config['kernel_size']

        image_input = layers.Input(self.image_size)
        img = self.conv_block(
            image_input,
            filters=disc_channels[0],
            kernel_size=kernel_size,
            strides=(
                2,
                2),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        
        for i in range(1, len(disc_channels)):
            img = self.conv_block(
                img,
                filters=disc_channels[i],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
        flatten = layers.Flatten()(img)
        valid = layers.Dense(1)(flatten)
        conv_out = self.conv_block(img,
                                   filters=disc_channels[-1],
                                   kernel_size=kernel_size,
                                   strides=(1,
                                            1),
                                   activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer)
        conv_out = layers.Flatten()(conv_out)
        discrete_out = layers.Dense(
            self.n_classes, activation="softmax")(conv_out)
        cont_out = layers.Dense(self.code_dim)(conv_out)
        disc_model = tf.keras.Model(inputs=image_input, outputs=[
                                    valid, discrete_out, cont_out])

        return disc_model

    def generator(self, config):

        n_classes = config['n_classes'] 
        noise_dim = config['noise_dim'] 
        code_dim = config['code_dim'] 

        
        dropout_rate = config['dropout_rate'] 
        gen_channels = config['gen_channels'] 
        activation = config['activation'] 
        kernel_initializer = config['kernel_initializer'] 
        kernel_regularizer = config['kernel_regularizer']
        kernel_size = config['kernel_size']

        input_shape = self.noise_dim + self.n_classes + self.code_dim
        input_noise = layers.Input(shape=input_shape)
        _input = layers.Dense((self.image_size[0] //
                               4) *
                              (self.image_size[1]) //
                              4 *
                              (gen_channels[0] *
                               2), use_bias=False)(input_noise)
        _input = layers.BatchNormalization()(_input)
        _input = layers.LeakyReLU()(_input)

        img = layers.Reshape(
            ((self.image_size[0] // 4),
             (self.image_size[1] // 4),
                (gen_channels[0] * 2)))(_input)

        for i in range(len(gen_channels)):
            img = self.conv_block(
                img,
                filters=gen_channels[i],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                conv_type="transpose",
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)

        img = self.conv_block(img, filters = self.image_size[-1], kernel_size=kernel_size, strides=(
            1, 1), use_batch_norm=False, activation="tanh", conv_type="transpose")

        gen_model = tf.keras.Model(input_noise, img)
        return gen_model
    
    def __load_model(self):

        self.gen_model, self.disc_model = self.generator(self.config), self.discriminator(self.config)

    def fit(self,
            train_ds=None,
            epochs=100,
            gen_optimizer='Adam',
            disc_optimizer='Adam',
            verbose=1,
            gen_learning_rate=0.0001,
            disc_learning_rate=0.0002,
            beta_1=0.5,
            tensorboard=False,
            save_model=None):

        assert train_ds is not None, 'No Input data found'
        
        self.__load_model()
        
        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        if(gen_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_learning_rate
        if(disc_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        disc_optimizer = getattr(tf.keras.optimizers, disc_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0
        generator_loss = tf.keras.metrics.Mean()
        discriminator_loss = tf.keras.metrics.Mean()
        
        total_batches = tf.data.experimental.cardinality(train_ds).numpy()
        
        for epoch in range(epochs):
            
            generator_loss.reset_states()
            discriminator_loss.reset_states()
            
            pbar = tqdm(total = total_batches, desc = 'Epoch - '+str(epoch+1))
            for data in train_ds:
                batch_size = data.shape[0]
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    Z = np.random.randn(batch_size, self.noise_dim)
                    label_input = tf.keras.utils.to_categorical(
                        (np.random.randint(0, self.n_classes, batch_size)), self.n_classes)
                    code_input = np.random.randn(batch_size, self.code_dim)
                    c = np.concatenate((Z, label_input, code_input), axis=1)

                    gen_imgs = self.gen_model(c, training=True)
                    real_output, _, _ = self.disc_model(data, training=True)
                    fake_output, discrete, cont_out = self.disc_model(
                        gen_imgs, training=True)

                    info_loss = auxillary_loss(
                        discrete, label_input, code_input, cont_out)
                    gen_loss = gan_generator_loss(fake_output) + info_loss
                    disc_loss = gan_discriminator_loss(
                        real_output, fake_output) + info_loss

                    generator_grads = gen_tape.gradient(
                        gen_loss, self.gen_model.trainable_variables)
                    discriminator_grads = disc_tape.gradient(
                        disc_loss, self.disc_model.trainable_variables)

                    gen_optimizer.apply_gradients(
                        zip(generator_grads, self.gen_model.trainable_variables))
                    disc_optimizer.apply_gradients(
                        zip(discriminator_grads, self.disc_model.trainable_variables))

                    generator_loss.update_state(gen_loss)
                    discriminator_loss.update_state(disc_loss)

                    pbar.update(1)
                    steps += 1

                    if(tensorboard):
                        with train_summary_writer.as_default():
                            tf.summary.scalar(
                                'discr_loss', disc_loss.numpy(), step=steps)
                            tf.summary.scalar(
                                'genr_loss', gen_loss.numpy(), step=steps)
            pbar.close()
            del pbar
            
            if(verbose):
                print('Epoch:',
                    epoch + 1,
                    'D_loss:',
                    generator_loss.result().numpy(),
                    'G_loss',
                    discriminator_loss.result().numpy())
            
        if(save_model is not None):

            assert isinstance(save_model, str), "Not a valid directory"
            if(save_model[-1] != '/'):
                self.gen_model.save_weights(
                    save_model + '/generator_checkpoint')
                self.disc_model.save_weights(
                    save_model + '/discriminator_checkpoint')
            else:
                self.gen_model.save_weights(
                    save_model + 'generator_checkpoint')
                self.disc_model.save_weights(
                    save_model + 'discriminator_checkpoint')

    def generate_samples(self, n_samples=1, save_dir=None):

        Z = np.random.randn(n_samples, self.noise_dim)
        label_input = tf.keras.utils.to_categorical(
            (np.random.randint(0, self.n_classes, n_samples)), self.n_classes)
        code_input = np.random.randn(n_samples, self.code_dim)

        seed = np.concatenate((Z, label_input, code_input), axis=1)
        generated_samples = self.gen_model(seed).numpy()

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
