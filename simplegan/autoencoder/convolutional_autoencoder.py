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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

### Silence Imageio warnings
def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


__all__ = ["ConvolutionalAutoencoder"]


class ConvolutionalAutoencoder:

    r"""Convolutional Autoencoder model

    Args:
        interm_dim (int, optional): represents the dimension of the bottleneck layer. Defaults to ``128``
        enc_channels (int, list, optional): represents the number of filters in the encoder part of the network. Defaults to ``[32, 64]``
        dec_channels (int, list, optional): represents the number of filters in the decoder part of the network. Defaults to ``[64, 32]```
        kernel_size (int, tuple, optional): repersents the size of the kernel to perform the convolution. Defaults to ``(5, 5)``
        activation (str, optional): type of non-linearity to be applied. Defaults to ``relu``
        kernel_initializer (str, optional): initialization of kernel weights. Defaults to ``glorot_uniform``
        kernel_regularizer (str, optional): type of regularization to be applied to the weights. Defaults to ``None``
    """

    def __init__(
        self,
        interm_dim=128,
        enc_channels=[32, 64],
        dec_channels=[64, 32],
        kernel_size=(5, 5),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
    ):

        self.model = tf.keras.Sequential()
        self.image_size = None
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
            two tensorflow dataset objects representing the train and test datset
        """

        if use_mnist:

            train_data, test_data = load_mnist_AE()

        elif use_cifar10:

            train_data, test_data = load_cifar10_AE()

        else:

            train_data, test_data = load_custom_data_AE(data_dir, img_shape)

        self.image_size = train_data.shape[1:]

        train_data = train_data / 255
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)
        )

        test_data = test_data / 255
        test_ds = (
            tf.data.Dataset.from_tensor_slices(test_data).shuffle(10000).batch(batch_size)
        )

        return train_ds, test_ds

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
            sample_images.append(img)

        sample_images = np.array(sample_images)

        if save_dir is None:
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)

    def encoder(self, config):

        enc_channels = config["enc_channels"]
        encoder_layers = len(enc_channels)
        interm_dim = config["interm_dim"]
        activation = config["activation"]
        kernel_initializer = config["kernel_initializer"]
        kernel_regularizer = config["kernel_regularizer"]
        kernel_size = config["kernel_size"]

        model = tf.keras.Sequential()

        model.add(
            Conv2D(
                enc_channels[0] // 2,
                kernel_size=kernel_size,
                padding="same",
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_shape=self.image_size,
            )
        )
        model.add(MaxPool2D())

        for i in range(encoder_layers):
            model.add(
                Conv2D(
                    enc_channels[i],
                    kernel_size=kernel_size,
                    padding="same",
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(MaxPool2D())

        model.add(Flatten())
        model.add(Dense(interm_dim, activation="sigmoid"))

        return model

    def decoder(self, config):

        dec_channels = config["dec_channels"]
        decoder_layers = len(dec_channels)
        interm_dim = config["interm_dim"]
        activation = config["activation"]
        kernel_initializer = config["kernel_initializer"]
        kernel_regularizer = config["kernel_regularizer"]
        kernel_size = config["kernel_size"]

        model = tf.keras.Sequential()

        model.add(
            Dense(
                (self.image_size[0] // 4) * (self.image_size[1] // 4) * (dec_channels[0] * 2),
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_dim=interm_dim,
            )
        )

        model.add(
            Reshape(
                ((self.image_size[0] // 4), (self.image_size[1] // 4), (dec_channels[0] * 2))
            )
        )

        k = 0
        for _ in range(decoder_layers // 2):
            model.add(
                Conv2DTranspose(
                    dec_channels[k],
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            k += 1

        model.add(
            Conv2DTranspose(
                dec_channels[k],
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        )

        for _ in range(decoder_layers // 2):
            model.add(
                Conv2DTranspose(
                    dec_channels[k],
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            k += 1

        model.add(
            Conv2DTranspose(
                self.image_size[2],
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                activation="tanh",
            )
        )

        return model

    def __load_model(self):

        self.model.add(self.encoder(self.config))
        self.model.add(self.decoder(self.config))

    def fit(
        self,
        train_ds=None,
        epochs=100,
        optimizer="Adam",
        verbose=1,
        learning_rate=0.001,
        tensorboard=False,
        save_model=None,
    ):

        r"""Function to train the model

        Args:
            train_ds (tf.data object): training data
            epochs (int, optional): number of epochs to train the model. Defaults to ``100``
            optimizer (str, optional): optimizer used to train the model. Defaults to ``Adam``
            verbose (int, optional): 1 - prints training outputs, 0 - no outputs. Defaults to ``1``
            learning_rate (float, optional): learning rate of the optimizer. Defaults to ``0.001``
            tensorboard (bool, optional): if true, writes loss values to ``logs/gradient_tape`` directory
                which aids visualization. Defaults to ``False``
            save_model (str, optional): Directory to save the trained model. Defaults to ``None``
        """

        assert train_ds is not None, "Initialize training data through train_ds parameter"

        self.__load_model()

        kwargs = {}
        kwargs["learning_rate"] = learning_rate
        optimizer = getattr(tf.keras.optimizers, optimizer)(**kwargs)

        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/" + current_time + "/train"
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0
        train_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(train_ds).numpy()
        except:
            total = 0

        for epoch in range(epochs):

            train_loss.reset_states()

            pbar = tqdm(total=total, desc="Epoch - " + str(epoch + 1))
            for data in train_ds:

                with tf.GradientTape() as tape:
                    recon_data = self.model(data)
                    loss = mse_loss(data, recon_data)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                train_loss(loss)

                steps += 1
                pbar.update(1)

                if tensorboard:
                    with train_summary_writer.as_default():
                        tf.summary.scalar("loss", loss.numpy(), step=steps)

            pbar.close()
            del pbar

            if verbose == 1:
                print("Epoch:", epoch + 1, "reconstruction loss:", train_loss.result().numpy())

        if save_model is not None:

            assert isinstance(save_model, str), "Not a valid directory"
            if save_model[-1] != "/":
                self.model.save_weights(
                    save_model + "/convolutional_autoencoder_autoencoder_checkpoint"
                )
            else:
                self.model.save_weights(
                    save_model + "convolutional_autoencoder_autoencoder_checkpoint"
                )

    def generate_samples(self, test_ds=None, save_dir=None):

        r"""Generate samples using the trained model

        Args:
            test_ds (tf.data object): test data object used to generate samples
            save_dir (str, optional): directory to save the generated images. Defaults to ``None``

        Return:
            returns ``None`` if save_dir is ``not None``, otherwise returns a numpy array with generated samples
        """

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = np.array([])
        for i, data in enumerate(test_ds):
            gen_sample = self.model(data, training=False)
            gen_sample = gen_sample.numpy()
            if i == 0:
                generated_samples = gen_sample
            else:
                generated_samples = np.concatenate((generated_samples, gen_sample), 0)

        if save_dir is None:
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)
