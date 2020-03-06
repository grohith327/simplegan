import os
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose, Dense, Reshape, Flatten
from tensorflow.keras import Model
from datasets.load_cifar10 import load_cifar10
from datasets.load_mnist import load_mnist
from datasets.load_custom_data import load_custom_data
from datasets.load_cifar100 import load_cifar100
from datasets.load_lsun import load_lsun
from losses.minmax_loss import gan_discriminator_loss, gan_generator_loss
import cv2
import numpy as np
import datetime
import tensorflow as tf
import sys
sys.path.append('..')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
DCGAN imports from tensorflow Model class

DCGAN paper: https://arxiv.org/abs/1511.06434
'''


class DCGAN():

    def __init__(self):

        self.image_size = None
        self.noise_dim = None
        self.gen_model = None
        self.disc_model = None

    def load_data(self, 
                data_dir=None, 
                use_mnist=False,
                use_cifar10=False, 
                use_cifar100=False, 
                use_lsun=False,
                batch_size=32, img_shape=(64, 64)):
        '''
        choose the dataset, if None is provided returns an assertion error -> ../datasets/load_custom_data
        returns a tensorflow dataset loader
        '''

        if(use_mnist):

            train_data = load_mnist()

        elif(use_cifar10):

            train_data = load_cifar10()

        elif(use_cifar100):

            train_data = load_cifar100()

        elif(use_lsun):

            train_data = load_lsun()

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
    Create a child class to modify generator and discriminator architecture for
    custom dataset
    '''

    def generator(self, params):

        noise_dim = params['noise_dim'] if 'noise_dim' in params else 100
        self.noise_dim = noise_dim
        gen_channels = params['gen_channels'] if 'gen_channels' in params else [
            64, 32, 16]
        gen_layers = params['gen_layers'] if 'gen_layers' in params else 3
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (
            5, 5)

        assert len(gen_channels) == gen_layers, "Dimension mismatch: length of generator channels should match number of generator layers"

        model = tf.keras.Sequential()
        model.add(
            Dense(
                (self.image_size[0] // 4) * (
                    self.image_size[1] // 4) * (
                    gen_channels[0] * 2),
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=noise_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(
            Reshape(
                ((self.image_size[0] // 4),
                 (self.image_size[1] // 4),
                    (gen_channels[0] * 2))))

        i = 0
        for _ in range(gen_layers // 2):
            model.add(
                Conv2DTranspose(
                    gen_channels[i],
                    kernel_size=kernel_size,
                    strides=(
                        1,
                        1),
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            model.add(BatchNormalization())
            model.add(LeakyReLU())
            i += 1

        model.add(
            Conv2DTranspose(
                gen_channels[i],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        for _ in range(gen_layers // 2):
            model.add(
                Conv2DTranspose(
                    gen_channels[i],
                    kernel_size=kernel_size,
                    strides=(
                        1,
                        1),
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            model.add(BatchNormalization())
            model.add(LeakyReLU())
            i += 1

        model.add(
            Conv2DTranspose(
                self.image_size[2],
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                padding='same',
                use_bias=False,
                activation='tanh'))

        return model

    def discriminator(self, params):

        dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 0.4
        disc_channels = params['disc_channels'] if 'disc_channels' in params else [
            16, 32, 64]
        disc_layers = params['disc_layers'] if 'disc_layers' in params else 3
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (
            5, 5)

        assert len(disc_channels) == disc_layers, "Dimension mismatch: length of discriminator channels should match number of discriminator layers"

        model = tf.keras.Sequential()

        model.add(
            Conv2D(
                disc_channels[0] // 2,
                kernel_size=kernel_size,
                strides=(
                    2,
                    2),
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_shape=self.image_size))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

        for i in range(disc_layers):
            model.add(
                Conv2D(
                    disc_channels[i],
                    kernel_size=kernel_size,
                    strides=(
                        1,
                        1),
                    padding='same',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            model.add(LeakyReLU())
            model.add(Dropout(dropout_rate))

        model.add(Conv2D(disc_channels[-1] * 2,
                         kernel_size=kernel_size,
                         strides=(2,
                                  2),
                         padding='same',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    '''
    call build_model() to get the generator and discriminator objects
    '''

    def build_model(
        self,
        params={
            'gen_layers': 3,
            'disc_layers': 3,
            'noise_dim': 100,
            'dropout_rate': 0.4,
            'activation': 'relu',
            'kernel_initializer': 'glorot_uniform',
            'kernel_size': (
            5,
            5),
            'gen_channels': [
                64,
                32,
                16],
            'disc_channels': [
            16,
            32,
            64],
            'kernel_regularizer': None}):

        self.gen_model, self.disc_model = self.generator(
            params), self.discriminator(params)

    def fit(self,
            train_ds=None,
            epochs=100,
            gen_optimizer='Adam',
            disc_optimizer='Adam',
            print_steps=100,
            gen_learning_rate=0.0001,
            disc_learning_rate=0.0002,
            beta_1=0.5,
            tensorboard=False,
            save_model=None):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'

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

        for epoch in range(epochs):
            for data in train_ds:

                with tf.GradientTape() as tape:

                    Z = tf.random.normal([data.shape[0], self.noise_dim])
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    real_logits = self.disc_model(data)
                    D_loss = gan_discriminator_loss(real_logits, fake_logits)

                gradients = tape.gradient(
                    D_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(
                    zip(gradients, self.disc_model.trainable_variables))

                with tf.GradientTape() as tape:

                    Z = tf.random.normal([data.shape[0], self.noise_dim])
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = gan_generator_loss(fake_logits)

                gradients = tape.gradient(
                    G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gradients, self.gen_model.trainable_variables))

                if(steps % print_steps == 0):
                    print(
                        'Step:',
                        steps + 1,
                        'D_loss:',
                        D_loss.numpy(),
                        'G_loss',
                        G_loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'discr_loss', D_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'genr_loss', G_loss.numpy(), step=steps)

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

        Z = tf.random.normal([n_samples, self.noise_dim])
        generated_samples = self.gen_model(Z).numpy()

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
