import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, Dense, Reshape, Flatten
from tensorflow.keras import Model
from ..datasets.load_cifar10 import load_cifar10
from ..datasets.load_mnist import load_mnist
from ..datasets.load_custom_data import load_custom_data
from ..datasets.load_cifar100 import load_cifar100
from .dcgan import DCGAN
from ..losses.wasserstein_loss import wgan_discriminator_loss, wgan_generator_loss
import cv2
import numpy as np
import datetime
from ..datasets.load_lsun import load_lsun
import imageio
from tqdm.auto import tqdm

### Silence Imageio warnings
def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

__all__ = ["WGAN"]

"""
References:
-> https://arxiv.org/abs/1701.07875
"""


class WGAN(DCGAN):

    r"""`WGAN <https://arxiv.org/abs/1701.07875>`_ model

    Args:
        noise_dim (int, optional): represents the dimension of the prior to sample values. Defaults to ``100``
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
        dropout_rate=0.4,
        gen_channels=[64, 32, 16],
        disc_channels=[16, 32, 64],
        kernel_size=(5, 5),
        activation="relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
    ):

        DCGAN.__init__(
            self,
            noise_dim,
            dropout_rate,
            gen_channels,
            disc_channels,
            kernel_size,
            activation,
            kernel_initializer,
            kernel_regularizer,
        )

    def __load_model(self):

        self.gen_model, self.disc_model = self.generator(), self.discriminator()

    def fit(
        self,
        train_ds=None,
        epochs=100,
        gen_optimizer="RMSprop",
        disc_optimizer="RMSprop",
        verbose=1,
        gen_learning_rate=5e-5,
        disc_learning_rate=5e-5,
        beta_1=0.5,
        tensorboard=False,
        save_model=None,
    ):

        r"""Function to train the model

        Args:
            train_ds (tf.data object): training data
            epochs (int, optional): number of epochs to train the model. Defaults to ``100``
            gen_optimizer (str, optional): optimizer used to train generator. Defaults to ``RMSprop``
            disc_optimizer (str, optional): optimizer used to train discriminator. Defaults to ``RMSprop``
            verbose (int, optional): 1 - prints training outputs, 0 - no outputs. Defaults to ``1``
            gen_learning_rate (float, optional): learning rate of the generator optimizer. Defaults to ``5e-5``
            disc_learning_rate (float, optional): learning rate of the discriminator optimizer. Defaults to ``5e-5``
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
            for data in train_ds:

                for _ in range(5):

                    with tf.GradientTape() as tape:

                        Z = tf.random.normal([data.shape[0], self.noise_dim])
                        fake = self.gen_model(Z)
                        fake_logits = self.disc_model(fake)
                        real_logits = self.disc_model(data)
                        D_loss = wgan_discriminator_loss(real_logits, fake_logits)

                    gradients = tape.gradient(D_loss, self.disc_model.trainable_variables)
                    clipped_gradients = [
                        (tf.clip_by_value(grad, -0.01, 0.01)) for grad in gradients
                    ]
                    disc_optimizer.apply_gradients(
                        zip(clipped_gradients, self.disc_model.trainable_variables)
                    )

                    discriminator_loss(D_loss)

                with tf.GradientTape() as tape:

                    Z = tf.random.normal([data.shape[0], self.noise_dim])
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = wgan_generator_loss(fake_logits)

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
