from ..layers import (
    SpectralNormalization,
    GenResBlock,
    SelfAttention,
    DiscResBlock,
    DiscOptResBlock,
)
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from ..datasets.load_cifar10 import load_cifar10_with_labels
from ..datasets.load_custom_data import load_custom_data_with_labels
from ..losses.hinge_loss import hinge_loss_generator, hinge_loss_discriminator
import datetime
from tqdm import tqdm
import logging
import imageio

logging.getLogger("tensorflow").setLevel(logging.ERROR)
# Silence Imageio warnings


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

__all__ = ["SAGAN"]


class Generator(tf.keras.Model):
    def __init__(self, n_classes, filters=64):
        super(Generator, self).__init__()
        self.filters = filters
        self.sn_linear = SpectralNormalization(tf.keras.layers.Dense(filters * 16 * 4 * 4))
        self.rs = tf.keras.layers.Reshape((4, 4, 16 * filters))
        self.res_block1 = GenResBlock(
            n_classes=n_classes, filters=filters * 16, spectral_norm=True
        )
        self.res_block2 = GenResBlock(
            n_classes=n_classes, filters=filters * 8, spectral_norm=True
        )
        self.res_block3 = GenResBlock(
            n_classes=n_classes, filters=filters * 4, spectral_norm=True
        )
        self.attn = SelfAttention(spectral_norm=True)
        self.res_block4 = GenResBlock(
            n_classes=n_classes, filters=filters * 2, spectral_norm=True
        )
        self.res_block5 = GenResBlock(n_classes=n_classes, filters=filters, spectral_norm=True)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.snconv2d1 = SpectralNormalization(
            tf.keras.layers.Conv2D(kernel_size=3, filters=3, strides=1, padding="same")
        )

    def call(self, inp, labels):
        x = self.sn_linear(inp)
        x = tf.reshape(x, (-1, 4, 4, self.filters * 16))
        x = self.res_block1(x, labels=labels)
        x = self.res_block2(x, labels=labels)
        x = self.res_block3(x, labels=labels)
        x = self.attn(x)
        x = self.res_block4(x, labels=labels)
        x = self.res_block5(x, labels=labels)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.snconv2d1(x)
        x = tf.nn.tanh(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, n_classes, filters=64):
        super(Discriminator, self).__init__()
        self.opt_block1 = DiscOptResBlock(filters=filters, spectral_norm=True)
        self.res_block1 = DiscResBlock(filters=filters * 2, spectral_norm=True)
        self.attn = SelfAttention(spectral_norm=True)
        self.res_block2 = DiscResBlock(filters=filters * 4, spectral_norm=True)
        self.res_block3 = DiscResBlock(filters=filters * 8, spectral_norm=True)
        self.res_block4 = DiscResBlock(filters=filters * 16, spectral_norm=True)
        self.res_block5 = DiscResBlock(
            filters=filters * 16, downsample=False, spectral_norm=True
        )

        self.sn_dense1 = SpectralNormalization(tf.keras.layers.Dense(1))
        self.sn_embedding = tf.keras.layers.Embedding(n_classes, filters * 16)

    def call(self, inp, labels):
        h0 = self.opt_block1(inp)
        h1 = self.res_block1(h0)
        h1 = self.attn(h1)
        h2 = self.res_block2(h1)
        h3 = self.res_block3(h2)
        h4 = self.res_block4(h3)
        h5 = self.res_block5(h4)
        h5 = tf.nn.relu(h5)
        h6 = tf.reduce_sum(h5, [1, 2])
        out = self.sn_dense1(h6)
        h_labels = self.sn_embedding(labels)
        out += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
        return out


class SAGAN:
    r"""`Self-Attention GAN <https://arxiv.org/abs/1805.08318>`_ model 

    Args:
        noise_dim (int, optional): represents the dimension of the prior to sample values. Defaults to ``128``
    """

    def __init__(
        self, noise_dim=128,
    ):

        self.image_size = None
        self.noise_dim = noise_dim
        self.n_classes = None
        self.config = locals()

    def load_data(
        self,
        data_dir=None,
        use_mnist=False,
        use_cifar10=False,
        batch_size=32,
        img_shape=(64, 64),
    ):

        if use_cifar10:

            train_data, train_labels = load_cifar10_with_labels()
            self.n_classes = 10

        else:
            train_data, train_labels = load_custom_data_with_labels(data_dir, img_shape)
            self.n_classes = np.unique(train_labels).shape[0]

        # Resize images tp 128x128
        def resize(image, label):
            image = tf.image.resize(image, [128, 128])
            return image, label

        train_data = (train_data / 255) * 2 - 1
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        self.image_size = (128, 128, train_data[0].shape[-1])
        train_ds = train_ds.map(resize)
        train_ds = train_ds.shuffle(10000).batch(batch_size)

        return train_ds

    def get_sample(self, data=None, n_samples=1, save_dir=None):
        r"""View sample of the data

        Args:
            data (tf.data object): dataset to load samples from
            n_samples (int, optional): number of samples to load. Defaults to ``1``
            save_dir (str, optional): directory to save the sample images. Defaults to ``None``

        Return:
            ``None`` if save_dir is ``not None``, otherwise returns numpy array of samples with shape (n_samples, img_shape)
        """

        assert data is not None, "Data not provided"

        sample_images = []
        data = data.unbatch()
        for img, label in data.take(n_samples):

            img = img.numpy()
            sample_images.append(img)

        sample_images = np.array(sample_images)

        if save_dir is None:
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)

    def generator(self):

        r"""Generator module for Self-Attention GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        return Generator(self.n_classes)

    def discriminator(self):

        r"""Discriminator module for Self-Attention GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        return Discriminator(self.n_classes)

    def __load_model(self):
        self.gen_model, self.disc_model = (
            Generator(self.n_classes),
            Discriminator(self.n_classes),
        )

    @tf.function
    def train_step(self, images, labels):

        with tf.GradientTape() as disc_tape:
            bs = images.shape[0]
            noise = tf.random.normal([bs, self.noise_dim])
            fake_labels = tf.convert_to_tensor(np.random.randint(0, self.n_classes, bs))

            generated_images = self.gen_model(noise, labels)

            real_output = self.disc_model(images, labels, training=True)
            fake_output = self.disc_model(generated_images, fake_labels, training=True)
            disc_loss = hinge_loss_discriminator(real_output, fake_output)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.disc_model.trainable_variables
            )
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.disc_model.trainable_variables)
            )

        with tf.GradientTape() as gen_tape:
            noise = tf.random.normal([bs, self.noise_dim])
            fake_labels = tf.random.uniform((bs,), 0, 10, dtype=tf.int32)
            generated_images = self.gen_model(noise, fake_labels)

            fake_output = self.disc_model(generated_images, fake_labels, training=False)
            gen_loss = hinge_loss_generator(fake_output)
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.gen_model.trainable_variables
            )
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.gen_model.trainable_variables)
            )
            train_stats = {
                "d_loss": disc_loss,
                "g_loss": gen_loss,
                "d_grads": gradients_of_discriminator,
                "g_grads": gradients_of_generator,
            }
            return train_stats

    def fit(
        self,
        train_ds=None,
        epochs=100,
        gen_optimizer="Adam",
        disc_optimizer="Adam",
        verbose=1,
        gen_learning_rate=1e-4,
        disc_learning_rate=4e-4,
        beta_1=0,
        beta_2=0.9,
        tensorboard=False,
        save_model=None,
    ):
        r"""Function to train the model

        Args:
            train_ds (tf.data object): training data
            epochs (int, optional): number of epochs to train the model. Defaults to ``100``
            gen_optimizer (str, optional): optimizer used to train generator. Defaults to ``Adam``
            disc_optimizer (str, optional): optimizer used to train discriminator. Defaults to ``Adam``
            verbose (int, optional): 1 - prints training outputs, 0 - no outputs. Defaults to ``1``
            gen_learning_rate (float, optional): learning rate of the generator optimizer. Defaults to ``0.0001``
            disc_learning_rate (float, optional): learning rate of the discriminator optimizer. Defaults to ``0.0002``
            beta_1 (float, optional): decay rate of the first momement. set if ``Adam`` optimizer is used. Defaults to ``0.5``
            beta_2 (float, optional): decay rate of the second momement. set if ``Adam`` optimizer is used. Defaults to ``0.5``
            tensorboard (bool, optional): if true, writes loss values to ``logs/gradient_tape`` directory
                which aids visualization. Defaults to ``False``
            save_model (str, optional): Directory to save the trained model. Defaults to ``None``
        """
        self.__load_model()

        kwargs = {}
        kwargs["learning_rate"] = gen_learning_rate
        if gen_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        self.generator_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = disc_learning_rate
        if disc_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        self.discriminator_optimizer = getattr(tf.keras.optimizers, disc_optimizer)(**kwargs)

        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/" + current_time + "/train"
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0
        average_generator_loss = tf.keras.metrics.Mean()
        average_discriminator_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(train_ds).numpy()
        except BaseException:
            total = 0

        for epoch in range(epochs):

            average_generator_loss.reset_states()
            average_discriminator_loss.reset_states()
            pbar = tqdm(total=total, desc="Epoch - " + str(epoch + 1))

            for i, batch in enumerate(train_ds):
                image_batch, label_batch = batch
                label_batch = tf.squeeze(label_batch)
                train_stats = self.train_step(image_batch, label_batch)

                G_loss = train_stats["g_loss"]
                D_loss = train_stats["d_loss"]
                average_generator_loss(G_loss)
                average_discriminator_loss(D_loss)

                steps += 1
                pbar.update(1)

                if tensorboard:
                    with train_summary_writer.as_default():
                        tf.summary.scalar("discr_loss", D_loss.numpy(), step=steps)
                        tf.summary.scalar("genr_loss", G_loss.numpy(), step=steps)

            pbar.close()
            del pbar

            if verbose == 1:
                print(
                    "Epoch:",
                    epoch + 1,
                    "D_loss:",
                    average_generator_loss.result().numpy(),
                    "G_loss",
                    average_discriminator_loss.result().numpy(),
                )

        if save_model is not None:
            assert isinstance(save_model, str), "Not a valid directory"
            if save_model[-1] != "/":
                self.gen_model.save_weights(save_model + "/generator_checkpoint")
                self.disc_model.save_weights(save_model + "/discriminator_checkpoint")
            else:
                self.gen_model.save_weights(save_model + "generator_checkpoint")
                self.disc_model.save_weights(save_model + "discriminator_checkpoint")

    def generate_samples(self, n_samples=1, labels_list=None, save_dir=None):
        r"""Generate samples using the trained model

        Args:
            n_samples (int, optional): number of samples to generate. Defaults to ``1``
            labels_list (int, list): list of labels representing the class of sample to generate
            save_dir (str, optional): directory to save the generated images. Defaults to ``None``

        Return:
            returns ``None`` if save_dir is ``not None``, otherwise returns a numpy array with generated samples
        """

        assert labels_list is not None, "Enter list of labels to condition the generator"
        assert (
            len(labels_list) == n_samples
        ), "Number of samples does not match length of labels list"

        Z = np.random.uniform(-1, 1, (n_samples, self.noise_dim))
        labels_list = np.array(labels_list)
        generated_samples = self.gen_model([Z, labels_list]).numpy()

        if save_dir is None:
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)
