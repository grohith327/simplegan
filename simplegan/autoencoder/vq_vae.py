import cv2
import os
from tensorflow.keras.layers import Dropout, BatchNormalization, Lambda
from tensorflow.keras.layers import Dense, Reshape, Input, ReLU, Conv2D
from tensorflow.keras.layers import Conv2DTranspose, Embedding, Flatten
from tensorflow.keras import Model
import imageio  
import numpy as np
from ..datasets.load_custom_data import load_custom_data_AE
from ..datasets.load_mnist import load_mnist_AE
from ..datasets.load_cifar10 import load_cifar10_AE
import datetime
from ..losses.mse_loss import mse_loss
import tensorflow as tf
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

'''
vector quantized vae

Reference: https://arxiv.org/abs/1711.00937


The code is inspired by the following sources:
-> https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
-> https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
-> https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
'''

__all__ = ['VQ_VAE']

class VectorQuantizer(Model):

    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        initializer = tf.keras.initializers.VarianceScaling(
            distribution='uniform')
        self.embedding = tf.Variable(
            initializer(
                shape=[
                    self.embedding_dim,
                    self.num_embeddings]),
            trainable=True)
        # Embedding(self.num_embeddings, self.embedding_dim,
        #                            embeddings_initializer=initializer)

    def call(self, x):

        # x = tf.transpose(x, perm=[0, 2, 3, 1])
        flat_x = tf.reshape(x, [-1, self.embedding_dim])

        distances = (
            tf.math.reduce_sum(
                flat_x**2,
                axis=1,
                keepdims=True) -
            2 *
            tf.linalg.matmul(
                flat_x,
                self.embedding) +
            tf.math.reduce_sum(
                self.embedding**2,
                axis=0,
                keepdims=True))

        encoding_indices = tf.math.argmax(-distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = tf.linalg.matmul(
            encodings, tf.transpose(self.embedding))
        quantized = tf.reshape(quantized, x.shape)

        e_latent_loss = tf.math.reduce_mean(
            (tf.stop_gradient(quantized) - x)**2)
        q_latent_loss = tf.math.reduce_mean(
            (quantized - tf.stop_gradient(x))**2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + tf.stop_gradient(quantized - x)
        avg_probs = tf.math.reduce_mean(encodings, axis=0)
        perplexity = tf.math.exp(- tf.math.reduce_sum(avg_probs *
                                                      tf.math.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class residual(Model):

    def __init__(self, num_hiddens, num_residual_layers,
                 num_residual_hiddens):
        super(residual, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self.relu = ReLU()
        self.conv1 = Conv2D(
            self.num_residual_hiddens, 
            activation='relu', 
            kernel_size=(
                3,
                3),
            strides=(
                1,
                1),
            padding='same')

        self.conv2 = Conv2D(
            self.num_hiddens, 
            kernel_size=(1, 1), 
            strides=(1, 1))

    def call(self, x):

        for _ in range(self.num_residual_layers):

            output = self.relu(x)
            output = self.conv1(output)
            output = self.conv2(output)

            x += output

        x = self.relu(x)
        return x


class encoder(Model):

    def __init__(self, config):
        super(encoder, self).__init__()

        self.num_hiddens = config['num_hiddens']
        self.num_residual_hiddens = config['num_residual_hiddens']
        self.num_residual_layers = config['num_residual_layers']

        self.conv1 = Conv2D(
            self.num_hiddens // 2,
            kernel_size=(
                4,
                4),
            strides=(
                2,
                2),
            activation='relu')

        self.conv2 = Conv2D(
            self.num_hiddens, 
            kernel_size=(
                4,
                4), 
            strides=(
                2,
                2), 
            activation='relu')

        self.conv3 = Conv2D(
            self.num_hiddens, 
            kernel_size=(
                3,
                3),
            strides=(
                1,
                1))

        self.residual_stack = residual(
            self.num_hiddens,
            self.num_residual_layers,
            self.num_residual_hiddens)

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_stack(x)

        return x


class decoder(Model):

    def __init__(self, config, image_size):
        super(decoder, self).__init__()

        self.num_hiddens = config['num_hiddens']
        self.num_residual_hiddens = config['num_residual_hiddens']
        self.num_residual_layers = config['num_residual_layers']

        self.conv1 = Conv2D(
            self.num_hiddens, 
            kernel_size=(
                3,
                3),
            strides=(
                1,
                1),
            padding='same')

        self.residual_stack = residual(
            self.num_hiddens,
            self.num_residual_layers,
            self.num_residual_hiddens)

        self.flatten = Flatten()

        self.dense1 = Dense(
            (image_size[0] // 4) * (image_size[1] // 4) * 128,
            activation='relu')

        self.reshape = Reshape(
            ((image_size[0] // 4), (image_size[1] // 4), 128))

        self.upconv1 = Conv2DTranspose(
            self.num_hiddens // 2,
            kernel_size=(
                4,
                4),
            strides=(
                2,
                2),
            activation='relu', padding='same')

        self.upconv2 = Conv2DTranspose(
            image_size[-1], kernel_size=(4, 4), strides=(2, 2), padding='same')

    def call(self, x):

        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.reshape(x)
        x = self.upconv1(x)
        x = self.upconv2(x)

        return x


class nn_model(Model):

    def __init__(self, config, image_size):
        super(nn_model, self).__init__()

        embedding_dim = config['embedding_dim']
        commiment_cost = config['commiment_cost']
        num_embeddings = config['num_embeddings']

        self.encoder = encoder(config)
        self.pre_vq_conv = Conv2D(
                            embedding_dim, 
                            kernel_size=(
                                1, 
                                1), 
                            strides=(
                                1, 
                                1))
        self.decoder = decoder(config, image_size)
        self.vq_vae = VectorQuantizer(
            num_embeddings,
            embedding_dim,
            commiment_cost)

    def call(self, x):

        output = self.encoder(x)
        output = self.pre_vq_conv(output)
        loss, quantized, perplexity, encodings = self.vq_vae(output)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity


class VQ_VAE:

    def __init__(self,
                num_hiddens = 128,
                num_residual_hiddens = 32,
                num_residual_layers = 2,
                num_embeddings = 512,
                embedding_dim = 64,
                commiment_cost = 0.25):


        self.image_size = None
        self.model = None
        self.data_var = None
        self.config = locals()

    def load_data(self, data_dir=None, use_mnist=False,
                  use_cifar10=False, batch_size=32, img_shape=(64, 64)):

        if(use_mnist):

            train_data, test_data = load_mnist_AE()

        elif(use_cifar10):

            train_data, test_data = load_cifar10_AE()

        else:

            train_data, test_data = load_custom_data_AE(data_dir, img_shape)

        self.image_size = train_data.shape[1:]
        self.data_var = np.var(train_data / 255)

        train_data = (train_data / 255.0) - 0.5
        train_ds = tf.data.Dataset.from_tensor_slices(
            train_data).shuffle(10000).batch(batch_size)

        test_data = (test_data / 255.0) - 0.5
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

    def __load_model(self):

        self.model = nn_model(self.config, self.image_size)

    def fit(self, 
            train_ds=None, 
            epochs=100, 
            optimizer='Adam', 
            verbose=1,
            learning_rate=3e-4, 
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
        total_loss = tf.keras.metrics.Mean()
        VecQuant_loss = tf.keras.metrics.Mean()
        reconstruction_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(train_ds).numpy()
        except:
            total = 0

        for epoch in range(epochs):

            total_loss.reset_states()
            reconstruction_loss.reset_states()
            VecQuant_loss.reset_states()

            pbar = tqdm(total = total, desc = 'Epoch - '+str(epoch+1))
            for data in train_ds:

                with tf.GradientTape() as tape:
                    vq_loss, data_recon, perplexity = self.model(data)
                    recon_err = mse_loss(data_recon, data) / self.data_var
                    loss = vq_loss + recon_err

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

                total_loss(loss)
                reconstruction_loss(recon_err)
                VecQuant_loss(vq_loss)

                steps += 1
                pbar.update(1)

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'vq_loss', 
                            vq_loss.numpy(), 
                            step=steps)
                        tf.summary.scalar(
                            'reconstruction_loss', 
                            recon_err.numpy(), 
                            step=steps)

            pbar.close()
            del pbar

            if(verbose == 1):
                print('Epoch:',
                    epoch + 1,
                    'total_loss:',
                    total_loss.result().numpy(),
                    'vq_loss:',
                    VecQuant_loss.result().numpy(),
                    'reconstruction loss:',
                    reconstruction_loss.result().numpy())

        if(save_model is not None):

            assert isinstance(save_model, str), "Not a valid directory"
            if(save_model[-1] != '/'):
                self.model.save_weights(save_model + '/vq_vae_checkpoint')
            else:
                self.model.save_weights(save_model + 'vq_vae_checkpoint')

    def generate_samples(self, test_ds=None, save_dir=None):

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = np.array([])
        for i, data in enumerate(test_ds):
            _, gen_sample, _ = self.model(data, training=False)
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
