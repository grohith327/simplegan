import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.layers import BatchNormalization, Conv3D
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU
from tensorflow.keras.layers import Reshape, Dense
from tensorflow.keras import Model
from tqdm.auto import tqdm
from ..datasets.load_off import load_vox_from_off
from ..losses.minmax_loss import gan_discriminator_loss, gan_generator_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

__all__ = ["VoxelGAN"]

"""
Reference:
-> http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
"""


class VoxelGAN:

    r"""Implementation of `3DGAN <http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf>`_ model

    Args:
        noise_dim (int, optional): represents the dimension of the prior to sample values. Defaults to ``200``
        gen_channels (int, list, optional): represents the number of filters in the generator network. Defaults to ``[512, 256, 128, 64]``
        disc_channels (int, list, optional): represents the number of filters in the discriminator network. Defaults to ``[64, 128, 256, 512]```
        kernel_size (int, tuple, optional): repersents the size of the kernel to perform the convolution. Defaults to ``(4, 4, 4)``
        activation (str, optional): type of non-linearity to be applied. Defaults to ``relu``
        kernel_initializer (str, optional): initialization of kernel weights. Defaults to ``glorot_uniform``
        kernel_regularizer (str, optional): type of regularization to be applied to the weights. Defaults to ``None`` 
    """

    def __init__(
        self,
        noise_dim=200,
        gen_channels=[512, 256, 128, 64],
        disc_channels=[64, 128, 256, 512],
        kernel_size=(4, 4, 4),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
    ):

        self.noise_dim = noise_dim
        self.gen_model = None
        self.disc_model = None
        self.side_length = None
        self.config = locals()

    def __is_power_of_two(self, x):

        return x and (not (x & (x - 1)))

    def load_data(self, data_dir=None, use_modelnet=False, batch_size=100, side_length=64):

        r"""Load data to train the voxelgan model

        Args:
            data_dir (str, optional): string representing the directory to load data from. Defaults to ``None``
            use_modelnet (bool, optional): use the ModelNet10 dataset to train the model. Defaults to ``False``
            batch_size (int, optional): mini batch size for training the model. Defaults to ``100``
            side_length (int, optional): Dimension of the voxelized data. Defaults to ``64``

        Return:
            a tensorflow dataset objects representing the training datset
        """

        assert self.__is_power_of_two(side_length), "side_length must be a power of 2"

        if use_modelnet:

            data_obj = load_vox_from_off(side_length=side_length)

        else:

            data_obj = load_vox_from_off(datadir=data_dir, side_length=side_length)

        self.side_length = side_length
        data_voxels = data_obj.load_data()
        voxel_ds = (
            tf.data.Dataset.from_tensor_slices(data_voxels).shuffle(10000).batch(batch_size)
        )
        return voxel_ds

    def get_sample(self, data, n_samples=1, plot=False):

        r"""get a sample of the data or visualize it by plotting the samples. For an interative visualization set ``n_samples = 1``

        Args:
            data (tf.data object): dataset to load samples from
            n_samples (int, optional): number of samples to load. Defaults to ``1``
            plot (bool, optional): whether to plot the data for visualization

        Return:
            ``None`` if ``plot`` is ``True`` else a numpy array of samples of shape ``(n_samples, side_length, side_length, side_length, 1)``
        """

        assert data is not None, "Data not provided"
        data.unbatch()
        sample_data = []
        for sample in data.take(n_samples):

            sample_data.append(sample.numpy()[0])

        sample_data = np.array(sample_data)

        if not plot:
            return sample_data

        flag = -1
        try:
            import plotly.graph_objects as go

            flag = 1
        except ModuleNotFoundError:
            pass

        if flag == -1 or n_samples > 1:

            fig, axs = plt.subplots(n_samples)
            for i in range(n_samples):

                x = sample_data[i].nonzero()[0]
                y = sample_data[i].nonzero()[1]
                z = sample_data[i].nonzero()[2]

                axs[i].scatter(x, y, z, c="black")

        else:

            x = sample_data[0].nonzero()[0]
            y = sample_data[0].nonzero()[1]
            z = sample_data[0].nonzero()[2]

            fig = go.Figure(
                data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=3))]
            )

            fig.show()

    def generator(self):

        r"""Generator module for VoxelGAN(3DGAN). Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        noise_dim = self.config["noise_dim"]
        gen_channels = self.config["gen_channels"]
        gen_layers = len(gen_channels)
        activation = self.config["activation"]
        kernel_initializer = self.config["kernel_initializer"]
        kernel_size = self.config["kernel_size"]
        kernel_regularizer = self.config["kernel_regularizer"]

        assert (
            2 ** (gen_layers + 2) == self.side_length
        ), "2^(Number of generator channels) must be equal to side_length / 4"

        model = tf.keras.Sequential()

        model.add(
            Dense(
                2 * 2 * 2,
                activation=activation,
                kernel_initializer=kernel_initializer,
                input_dim=noise_dim,
            )
        )
        model.add(BatchNormalization())
        model.add(Reshape((2, 2, 2, 1)))

        for channel in gen_channels:
            model.add(
                Conv3DTranspose(
                    channel,
                    kernel_size,
                    strides=2,
                    padding="same",
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(BatchNormalization())

        model.add(
            Conv3DTranspose(1, kernel_size, strides=2, padding="same", activation="sigmoid")
        )

        return model

    def discriminator(self):

        r"""Discriminator module for VoxelGAN(3DGAN). Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        disc_channels = self.config["disc_channels"]
        disc_layers = len(disc_channels)
        kernel_initializer = self.config["kernel_initializer"]
        kernel_regularizer = self.config["kernel_regularizer"]
        kernel_size = self.config["kernel_size"]

        assert (
            2 ** (disc_layers + 2) == self.side_length
        ), "2^(Number of discriminator channels) must be equal to side_length / 4"

        model = tf.keras.Sequential()

        model.add(
            Conv3D(
                disc_channels[0],
                kernel_size=kernel_size,
                strides=2,
                padding="same",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                input_shape=(self.side_length, self.side_length, self.side_length, 1),
            )
        )
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        for channel in disc_channels[1:]:
            model.add(
                Conv3D(
                    channel,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(BatchNormalization())
            model.add(LeakyReLU())

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
        gen_learning_rate=0.0025,
        disc_learning_rate=0.00001,
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
            gen_learning_rate (float, optional): learning rate of the generator optimizer. Defaults to ``0.0025``
            disc_learning_rate (float, optional): learning rate of the discriminator optimizer. Defaults to ``0.00001``
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
        except BaseException:
            total = 0

        for epoch in range(epochs):

            generator_loss.reset_states()
            discriminator_loss.reset_states()

            pbar = tqdm(total=total, desc="Epoch - " + str(epoch + 1))
            for data in train_ds:

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(0, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    real_logits = self.disc_model(data)
                    D_loss = gan_discriminator_loss(real_logits, fake_logits)

                discriminator_loss(D_loss)

                if discriminator_loss.result().numpy() < 0.8:

                    gradients = tape.gradient(D_loss, self.disc_model.trainable_variables)
                    disc_optimizer.apply_gradients(
                        zip(gradients, self.disc_model.trainable_variables)
                    )

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(0, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = gan_generator_loss(fake_logits)

                gradients = tape.gradient(G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gradients, self.gen_model.trainable_variables)
                )

                generator_loss(G_loss)

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

    def generate_sample(self, n_samples=1, plot=False):
        # For optimal viewing, set n_samples to 1

        r"""Generate samples from the trained model and visualize them. For an interative visualization set ``n_samples = 1``

        Args:
            n_samples (int, optional): number of samples to generate. Defaults to ``1``
            plot (bool, optional): whether to plot the data for visualization

        Return:
            ``None`` if ``plot`` is ``True`` else a numpy array of samples of shape ``(n_samples, side_length, side_length, side_length, 1)``
        """

        Z = np.random.uniform(0, 1, (n_samples, self.noise_dim))
        generated_samples = self.gen_model(Z).numpy()

        if not plot:
            return generated_samples

        flag = -1
        try:
            import plotly.graph_objects as go

            flag = 1
        except ModuleNotFoundError:
            pass

        if flag == -1 or n_samples > 1:

            fig, axs = plt.subplots(n_samples)
            for i in range(n_samples):

                x = generated_samples[i].nonzero()[0]
                y = generated_samples[i].nonzero()[1]
                z = generated_samples[i].nonzero()[2]

                axs[i].scatter(x, y, z, c="black")

        else:

            x = generated_samples[0].nonzero()[0]
            y = generated_samples[0].nonzero()[1]
            z = generated_samples[0].nonzero()[2]

            fig = go.Figure(
                data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=3))]
            )

            fig.show()
