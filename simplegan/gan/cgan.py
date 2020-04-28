import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from ..datasets.load_mnist import load_mnist_with_labels
from ..datasets.load_cifar10 import load_cifar10_with_labels
from ..datasets.load_custom_data import load_custom_data_with_labels
from ..losses.minmax_loss import gan_discriminator_loss, gan_generator_loss
import datetime
import imageio
from tqdm.auto import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

### Silence Imageio warnings
def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

"""
References:
-> https://arxiv.org/abs/1411.1784
"""

__all__ = ["CGAN"]


class CGAN:

    r"""`CGAN <https://arxiv.org/abs/1411.1784>`_ model

    Args:
        noise_dim (int, optional): represents the dimension of the prior to sample values. Defaults to ``100``
        embed_dim (int, optional): represents the dimension of the embedding layer to convert classes to dense features. Defaults to ``100``
        dropout_rate (float, optional): represents the amount of dropout regularization to be applied. Defaults to ``0.4``
        gen_channels (int, list, optional): represents the number of filters in the generator network. Defaults to ``[64, 32, 16]``
        disc_channels (int, list, optional): represents the number of filters in the discriminator network. Defaults to ``[16, 32, 64]```
        kernel_size (int, tuple, optional): repersents the size of the kernel to perform the convolution. Defaults to ``(5, 5)``
        activation (str, optional): type of non-linearity to be applied. Defaults to ``relu``
        kernel_initializer (str, optional): initialization of kernel weights. Defaults to ``glorot_uniform``
        kernel_regularizer (str, optional): type of regularization to be applied to the weights. Defaults to ``None``
    """

    def __init__(
        self,
        noise_dim=100,
        embed_dim=100,
        dropout_rate=0.4,
        gen_channels=[64, 32, 16],
        disc_channels=[16, 32, 64],
        kernel_size=(5, 5),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
    ):

        self.image_size = None
        self.embed_dim = embed_dim
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

        r"""Load data to train the model

        Args:
            data_dir (str, optional): string representing the directory to load data from. Defaults to ``None``
            use_mnist (bool, optional): use the MNIST dataset to train the model. Defaults to ``False``
            use_cifar10 (bool, optional): use the CIFAR10 dataset to train the model. Defaults to ``False``
            use_lsun (bool, optional): use the LSUN dataset to train the model. Defaults to ``False``
            batch_size (int, optional): mini batch size for training the model. Defaults to ``32``
            img_shape (int, tuple, optional): shape of the image when loading data from custom directory. Defaults to ``(64, 64)``

        Return:
            a tensorflow dataset objects representing the training datset
        """

        if use_mnist:

            train_data, train_labels = load_mnist_with_labels()
            self.n_classes = 10

        elif use_cifar10:

            train_data, train_labels = load_cifar10_with_labels()
            self.n_classes = 10

        else:

            train_data, train_labels = load_custom_data_with_labels(data_dir, img_shape)
            self.n_classes = np.unique(train_labels).shape[0]

        self.image_size = train_data[0].shape

        train_data = (train_data - 127.5) / 127.5
        train_ds = (
            tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            .shuffle(10000)
            .batch(batch_size)
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

        r"""Generator module for Conditional GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        gen_channels = self.config["gen_channels"]
        gen_layers = len(gen_channels)
        activation = self.config["activation"]
        kernel_initializer = self.config["kernel_initializer"]
        kernel_regularizer = self.config["kernel_regularizer"]
        kernel_size = self.config["kernel_size"]

        z = layers.Input(shape=self.noise_dim)
        label = layers.Input(shape=1)

        start_image_size = (self.image_size[0] // 4, self.image_size[1] // 4)

        embedded_label = layers.Embedding(input_dim=self.n_classes, output_dim=self.embed_dim)(
            label
        )
        embedded_label = layers.Dense(
            units=start_image_size[0] * start_image_size[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            input_dim=self.embed_dim,
        )(embedded_label)
        embedded_label = layers.Reshape((start_image_size[0], start_image_size[1], 1))(
            embedded_label
        )

        input_img = layers.Dense(
            start_image_size[0] * start_image_size[1] * gen_channels[0] * 2,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )(z)
        input_img = layers.Reshape(
            (start_image_size[0], start_image_size[1], gen_channels[0] * 2)
        )(input_img)

        x = layers.Concatenate()([input_img, embedded_label])

        for i in range(gen_layers):
            x = layers.Conv2DTranspose(
                filters=gen_channels[i],
                kernel_size=kernel_size,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(
            filters=gen_channels[-1] // 2,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        output = layers.Conv2DTranspose(
            filters=self.image_size[-1],
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )(x)
        model = tf.keras.Model([z, label], output)
        return model

    def discriminator(self):

        r"""Discriminator module for Conditional GAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        dropout_rate = self.config["dropout_rate"]
        disc_channels = self.config["disc_channels"]
        disc_layers = len(disc_channels)
        activation = self.config["activation"]
        kernel_initializer = self.config["kernel_initializer"]
        kernel_regularizer = self.config["kernel_regularizer"]
        kernel_size = self.config["kernel_size"]

        input_image = layers.Input(shape=self.image_size)
        input_label = layers.Input(shape=1)

        embedded_label = layers.Embedding(input_dim=self.n_classes, output_dim=self.embed_dim)(
            input_label
        )
        embedded_label = layers.Dense(
            units=self.image_size[0] * self.image_size[1], activation=activation
        )(embedded_label)
        embedded_label = layers.Reshape((self.image_size[0], self.image_size[1], 1))(
            embedded_label
        )

        x = layers.Concatenate()([input_image, embedded_label])

        for i in range(disc_layers):
            x = layers.Conv2D(
                filters=disc_channels[i],
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)
        fe = layers.Dropout(dropout_rate)(x)
        out_layer = layers.Dense(1, activation="sigmoid")(fe)

        model = tf.keras.Model(inputs=[input_image, input_label], outputs=out_layer)
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
        disc_learning_rate=0.0002,
        beta_1=0.5,
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
            tensorboard (bool, optional): if true, writes loss values to ``logs/gradient_tape`` directory
                which aids visualization. Defaults to ``False``
            save_model (str, optional): Directory to save the trained model. Defaults to ``None``
        """

        assert train_ds is not None, "Initialize training data through train_ds parameter"

        self.__load_model()

        kwargs = {}
        kwargs["learning_rate"] = gen_learning_rate
        if gen_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = disc_learning_rate
        if disc_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
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
            for data, labels in train_ds:
                sampled_labels = np.random.randint(0, 10, data.shape[0]).reshape(-1, 1)
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    noise = tf.random.normal([data.shape[0], self.noise_dim])
                    fake_imgs = self.gen_model([noise, sampled_labels])

                    real_output = self.disc_model([data, labels], training=True)
                    fake_output = self.disc_model([fake_imgs, sampled_labels], training=True)

                    G_loss = gan_generator_loss(fake_output)
                    D_loss = gan_discriminator_loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(
                    G_loss, self.gen_model.trainable_variables
                )
                gradients_of_discriminator = disc_tape.gradient(
                    D_loss, self.disc_model.trainable_variables
                )

                gen_optimizer.apply_gradients(
                    zip(gradients_of_generator, self.gen_model.trainable_variables)
                )
                disc_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, self.disc_model.trainable_variables)
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
