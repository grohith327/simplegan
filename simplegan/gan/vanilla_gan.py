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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

### Silence Imageio warnings
def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

"""
References: 
-> https://arxiv.org/abs/1406.2661
"""

__all__ = ["VanillaGAN"]


class VanillaGAN:

    r"""`Vanilla GAN <https://arxiv.org/abs/1406.2661>`_ model

    Args:
        noise_dim (int, optional): represents the dimension of the prior to sample values. Defaults to ``64``
        dropout_rate (float, optional): represents the amount of dropout regularization to be applied. Defaults to ``0.4``
        gen_units (int, list, optional): represents the number of units/neurons in the generator network. Defaults to ``[128, 256, 512]``
        disc_units (int, list, optional): represents the number of units/neurons in the discriminator network. Defaults to ``[512, 256, 128]```
        activation (str, optional): type of non-linearity to be applied. Defaults to ``relu``
        kernel_initializer (str, optional): initialization of kernel weights. Defaults to ``glorot_uniform``
        kernel_regularizer (str, optional): type of regularization to be applied to the weights. Defaults to ``None``
    """

    def __init__(
        self,
        noise_dim=64,
        dropout_rate=0.4,
        gen_units=[128, 256, 512],
        disc_units=[512, 256, 128],
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
    ):

        self.image_size = None
        self.noise_dim = noise_dim
        self.gen_model = None
        self.disc_model = None
        self.config = locals()

    def load_data(
        self,
        data_dir=None,
        use_mnist=False,
        use_cifar10=False,
        batch_size=32,
        img_shape=(64, 64),
    ):

        r"""Load data to train the model

        Args:
            data_dir (str, optional): string representing the directory to load data from. Defaults to ``None``
            use_mnist (bool, optional): use the MNIST dataset to train the model. Defaults to ``False``
            use_cifar10 (bool, optional): use the CIFAR10 dataset to train the model. Defaults to ``False``
            batch_size (int, optional): mini batch size for training the model. Defaults to ``32``
            img_shape (int, tuple, optional): shape of the image when loading data from custom directory. Defaults to ``(64, 64)``

        Return:
            a tensorflow dataset objects representing the training datset
        """

        if use_mnist:

            train_data = load_mnist()

        elif use_cifar10:

            train_data = load_cifar10()

        else:

            train_data = load_custom_data(data_dir, img_shape)

        self.image_size = train_data.shape[1:]

        train_data = (
            train_data.reshape(
                (-1, self.image_size[0] * self.image_size[1] * self.image_size[2])
            )
            / 255
        )
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)
        )

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
        for img in data.take(n_samples):

            img = img.numpy()
            img = img.reshape((self.image_size[0], self.image_size[1], self.image_size[2]))
            sample_images.append(img)

        sample_images = np.array(sample_images)

        if save_dir is None:
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)

    def generator(self):

        r"""Generator module for Vanilla GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        noise_dim = self.config["noise_dim"]
        dropout_rate = self.config["dropout_rate"]
        gen_units = self.config["gen_units"]
        gen_layers = len(gen_units)
        activation = self.config["activation"]
        kernel_initializer = self.config["kernel_initializer"]
        kernel_regularizer = self.config["kernel_regularizer"]

        model = tf.keras.Sequential()

        model.add(
            Dense(
                gen_units[0] // 2,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=noise_dim,
                dtype=tf.float32,
            )
        )
        model.add(Dropout(dropout_rate))

        for i in range(gen_layers):
            model.add(
                Dense(
                    gen_units[i],
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    dtype=tf.float32,
                )
            )
            model.add(Dropout(dropout_rate))

        model.add(
            Dense(
                self.image_size[0] * self.image_size[1] * self.image_size[2],
                activation="sigmoid",
                dtype=tf.float32,
            )
        )
        return model

    def discriminator(self):

        r"""Discriminator module for Vanilla GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        dropout_rate = self.config["dropout_rate"]
        disc_units = self.config["disc_units"]
        disc_layers = len(disc_units)
        activation = self.config["activation"]
        kernel_initializer = self.config["kernel_initializer"]
        kernel_regularizer = self.config["kernel_regularizer"]

        model = tf.keras.Sequential()

        model.add(
            Dense(
                disc_units[0] * 2,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=self.image_size[0] * self.image_size[1] * self.image_size[2],
            )
        )
        model.add(Dropout(dropout_rate))

        for i in range(disc_layers):
            model.add(
                Dense(
                    disc_units[i],
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))
        return model

    def __load_model(self):

        self.gen_model, self.disc_model = self.generator(), self.discriminator()

    def fit(
        self,
        train_ds=None,
        epochs=100,
        gen_optimizer="Adam",
        disc_optimizer="Adam",
        verbose=1,
        gen_learning_rate=0.0001,
        disc_learning_rate=0.0001,
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
            disc_learning_rate (float, optional): learning rate of the discriminator optimizer. Defaults to ``0.0001``
            tensorboard (bool, optional): if true, writes loss values to ``logs/gradient_tape`` directory
                which aids visualization. Defaults to ``False``
            save_model (str, optional): Directory to save the trained model. Defaults to ``None``
        """

        assert train_ds is not None, "Initialize training data through train_ds parameter"

        self.__load_model()

        kwargs = {}
        kwargs["learning_rate"] = gen_learning_rate
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = disc_learning_rate
        disc_optimizer = getattr(tf.keras.optimizers, disc_optimizer)(**kwargs)

        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/" + current_time + "/train"
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

            pbar = tqdm(total=total, desc="Epoch - " + str(epoch + 1))
            for data in train_ds:

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    real_logits = self.disc_model(data)
                    D_loss = gan_discriminator_loss(real_logits, fake_logits)

                gradients = tape.gradient(D_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(
                    zip(gradients, self.disc_model.trainable_variables)
                )

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = gan_generator_loss(fake_logits)

                gradients = tape.gradient(G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gradients, self.gen_model.trainable_variables)
                )

                generator_loss(G_loss)
                discriminator_loss(D_loss)

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
                    generator_loss.result().numpy(),
                    "G_loss",
                    discriminator_loss.result().numpy(),
                )

        if save_model is not None:

            assert isinstance(save_model, str), "Not a valid directory"
            if save_model[-1] != "/":
                self.gen_model.save_weights(save_model + "/generator_checkpoint")
                self.disc_model.save_weights(save_model + "/discriminator_checkpoint")
            else:
                self.gen_model.save_weights(save_model + "generator_checkpoint")
                self.disc_model.save_weights(save_model + "discriminator_checkpoint")

    def generate_samples(self, n_samples=1, save_dir=None):

        r"""Generate samples using the trained model

        Args:
            n_samples (int, optional): number of samples to generate. Defaults to ``1``
            save_dir (str, optional): directory to save the generated images. Defaults to ``None``

        Return:
            returns ``None`` if save_dir is ``not None``, otherwise returns a numpy array with generated samples
        """

        Z = np.random.uniform(-1, 1, (n_samples, self.noise_dim))
        generated_samples = self.gen_model(Z)
        generated_samples = tf.reshape(
            generated_samples,
            [n_samples, self.image_size[0], self.image_size[1], self.image_size[2]],
        ).numpy()

        if save_dir is None:
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)
