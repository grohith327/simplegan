import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from ..datasets.load_cifar10 import load_cifar10
from ..datasets.load_mnist import load_mnist
from ..datasets.load_custom_data import load_custom_data
from ..losses.minmax_loss import gan_discriminator_loss, gan_generator_loss
import numpy as np
import datetime
import cv2
import tensorflow as tf
import imageio
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

'''
vanilla gan imports from tensorflow Model class

Original GAN paper: https://arxiv.org/abs/1406.2661
'''

__all__ = ['VanillaGAN']


class VanillaGAN():

    def __init__(self,
                noise_dim = 64,
                dropout_rate = 0.4,
                activation = 'relu',
                kernel_initializer = 'glorot_uniform',
                gen_units = [
                    128,
                    256,
                    512],
                disc_units = [
                    512,
                    256,
                    128],
                kernel_regularizer = None):

        self.image_size = None
        self.noise_dim = noise_dim
        self.gen_model = None
        self.disc_model = None
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

    '''
    Create a child class to modify generator and discriminator architecture for
    custom dataset
    '''

    def generator(self, config):

        noise_dim = config['noise_dim']
        dropout_rate = config['dropout_rate']
        gen_units = config['gen_units']
        gen_layers = len(gen_units)
        activation = config['activation']
        kernel_initializer = config['kernel_initializer']
        kernel_regularizer = config['kernel_regularizer']

        model = tf.keras.Sequential()

        model.add(
            Dense(
                gen_units[0] // 2,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=noise_dim,
                dtype=tf.float32))
        model.add(Dropout(dropout_rate))

        for i in range(gen_layers):
            model.add(
                Dense(
                    gen_units[i],
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    dtype=tf.float32))
            model.add(Dropout(dropout_rate))

        model.add(
            Dense(
                self.image_size[0] *
                self.image_size[1] *
                self.image_size[2],
                activation='sigmoid',
                dtype=tf.float32))
        return model

    def discriminator(self, config):

        dropout_rate = config['dropout_rate']
        disc_units = config['disc_units']
        disc_layers = len(disc_units)
        activation = config['activation']
        kernel_initializer = config['kernel_initializer']
        kernel_regularizer = config['kernel_regularizer']


        model = tf.keras.Sequential()

        model.add(
            Dense(
                disc_units[0] *
                2,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=self.image_size[0] *
                self.image_size[1] *
                self.image_size[2]))
        model.add(Dropout(dropout_rate))

        for i in range(disc_layers):
            model.add(
                Dense(
                    disc_units[i],
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))
        return model

    '''
    call build_model() to get the generator and discriminator objects
    '''

    def __load_model(self):

        self.gen_model, self.disc_model = self.generator(
            self.config), self.discriminator(self.config)

    def fit(self,
            train_ds=None,
            epochs=100,
            gen_optimizer='Adam',
            disc_optimizer='Adam',
            verbose=1,
            gen_learning_rate=0.0001,
            disc_learning_rate=0.0001,
            tensorboard=False,
            save_model=None):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'

        self.__load_model()

        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_learning_rate
        disc_optimizer = getattr(tf.keras.optimizers, disc_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0
        generator_loss = tf.keras.metrics.Mean()
        discriminator_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(train_ds).numpy()
        except:
            total = 0

        for epoch in range(epochs):
            
            generator_loss.reset_states()
            discriminator_loss.reset_states()

            pbar = tqdm(total = total, desc = 'Epoch - '+str(epoch+1))
            for data in train_ds:

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1,
                                          (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    real_logits = self.disc_model(data)
                    D_loss = gan_discriminator_loss(real_logits, fake_logits)

                gradients = tape.gradient(
                    D_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(
                    zip(gradients, self.disc_model.trainable_variables))

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1,
                                          (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = gan_generator_loss(fake_logits)

                gradients = tape.gradient(
                    G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gradients, self.gen_model.trainable_variables))

                generator_loss(G_loss)
                discriminator_loss(D_loss)

                steps += 1
                pbar.update(1)

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'discr_loss', D_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'genr_loss', G_loss.numpy(), step=steps)


            pbar.close()
            del pbar

            if(verbose == 1):
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

        Z = np.random.uniform(-1, 1, (n_samples, self.noise_dim))
        generated_samples = self.gen_model(Z)
        generated_samples = tf.reshape(
            generated_samples, [
                n_samples, self.image_size[0], self.image_size[1], self.image_size[2]]).numpy()

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
