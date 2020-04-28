import os
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense, Reshape, Flatten, ReLU
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Model
from ..losses.minmax_loss import gan_generator_loss, gan_discriminator_loss
from ..losses.cyclegan_loss import cycle_loss, identity_loss
from ..datasets.load_cyclegan_datasets import cyclegan_dataloader
from .pix2pix import Pix2Pix
import tensorflow as tf
import numpy as np
import datetime
import cv2
import imageio
from tqdm.auto import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

### Silence Imageio warnings
def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

__all__ = ["CycleGAN"]

"""

References:
-> https://arxiv.org/abs/1703.10593
-> https://www.tensorflow.org/tutorials/generative/cyclegan

"""


class CycleGAN(Pix2Pix):

    r"""`CycleGAN <https://arxiv.org/abs/1703.10593>`_ model. During training, samples are saved at ./samples and rate specified by save_img_per_epoch

    Args:
        dropout_rate (float, optional): represents the amount of dropout regularization to be applied. Defaults to ``0.5``
        gen_enc_channels (int, list, optional): represents the number of filters in the encoder part of Unet generator network. Defaults to ``[128, 256, 512,
            512, 512, 512, 512]``
        gen_dec_channels (int, list, optional): represents the number of filters in the decoder part of Unet generator network. Defaults to ``[512, 512, 512, 512,
            256, 128, 64]``
        disc_channels (int, list, optional): represents the number of filters in the discriminator network. Defaults to ``[64, 128, 256, 512]```
        kernel_size (int, tuple, optional): repersents the size of the kernel to perform the convolution. Defaults to ``(4, 4)``
        kernel_initializer (str, optional): initialization of kernel weights. Defaults to ``tf.random_normal_initializer(0., 0.02)``
    """

    def __init__(
        self,
        dropout_rate=0.5,
        gen_enc_channels=[128, 256, 512, 512, 512, 512, 512],
        gen_dec_channels=[512, 512, 512, 512, 256, 128, 64],
        disc_channels=[64, 128, 256, 512],
        kernel_size=(4, 4),
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
    ):

        Pix2Pix.__init__(
            self,
            dropout_rate,
            gen_enc_channels,
            gen_dec_channels,
            disc_channels,
            kernel_size,
            kernel_initializer,
        )

        self.gen_model_g = None
        self.gen_model_f = None
        self.disc_model_x = None
        self.disc_model_y = None

    def load_data(
        self,
        data_dir=None,
        use_apple2orange=False,
        use_summer2winter_yosemite=False,
        use_horse2zebra=False,
        use_monet2photo=False,
        use_cezanne2photo=False,
        use_ukiyoe2photo=False,
        use_vangogh2photo=False,
        use_maps=False,
        use_cityscapes=False,
        use_facades=False,
        use_iphone2dslr_flower=False,
        batch_size=32,
    ):

        r"""Load data to train the model

        Args:
            data_dir (str, optional): string representing the directory to load data from. Defaults to ``None``
            use_apple2orange (bool, optional): use the apple2orange dataset to train the model. Defaults to ``False``
            use_summer2winter_yosemite (bool, optional): use the summer2winter_yosemite dataset to train the model. Defaults to ``False``
            use_horse2zebra (bool, optional): use the horse2zebra dataset to train the model. Defaults to ``False``
            use_monet2photo (bool, optional): use the monet2photo dataset to train the model. Defaults to ``False``
            use_cezanne2photo (bool, optional): use the cezanne2photo dataset to train the model. Defaults to ``False``
            use_ukiyoe2photo (bool, optional): use the ukiyoe2photo dataset to train the model. Defaults to ``False``
            use_vangogh2photo (bool, optional): use the vangogh2photo dataset to train the model. Defaults to ``False``
            use_maps (bool, optional): use the maps dataset to train the model. Defaults to ``False``
            use_cityscapes (bool, optional): use the cityscapes dataset to train the model. Defaults to ``False``
            use_facades (bool, optional): use the facades dataset to train the model. Defaults to ``False``
            use_iphone2dslr_flower (bool, optional): use the iphone2dslr_flower dataset to train the model. Defaults to ``False``
            batch_size (int, optional): mini batch size for training the model. Defaults to ``32``

        Return:
            four tensorflow dataset objects representing trainA, trainB, testA, testB 
        """

        if use_apple2orange:

            data_obj = cyclegan_dataloader(dataset_name="apple2orange")

        elif use_summer2winter_yosemite:

            data_obj = cyclegan_dataloader(dataset_name="summer2winter_yosemite")

        elif use_horse2zebra:

            data_obj = cyclegan_dataloader(dataset_name="horse2zebra")

        elif use_monet2photo:

            data_obj = cyclegan_dataloader(dataset_name="monet2photo")

        elif use_cezanne2photo:

            data_obj = cyclegan_dataloader(dataset_name="cezanne2photo")

        elif use_ukiyoe2photo:

            data_obj = cyclegan_dataloader(dataset_name="ukiyoe2photo")

        elif use_vangogh2photo:

            data_obj = cyclegan_dataloader(dataset_name="vangogh2photo")

        elif use_maps:

            data_obj = cyclegan_dataloader(dataset_name="maps")

        elif use_cityscapes:

            data_obj = cyclegan_dataloader(dataset_name="cityscapes")

        elif use_facades:

            data_obj = cyclegan_dataloader(dataset_name="facades")

        elif use_iphone2dslr_flower:

            data_obj = cyclegan_dataloader(dataset_name="iphone2dslr_flower")

        else:

            data_obj = cyclegan_dataloader(datadir=data_dir)

        trainA, trainB, testA, testB = data_obj.load_dataset()

        for data in trainA.take(1):
            self.img_size = data.shape
            self.channels = data.shape[-1]

        trainA = trainA.shuffle(100000).batch(batch_size)
        trainB = trainB.shuffle(100000).batch(batch_size)

        testA = testA.shuffle(100000).batch(batch_size)
        testB = testB.shuffle(100000).batch(batch_size)

        return trainA, trainB, testA, testB

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
        data.unbatch()
        for img in data.take(n_samples):

            img = img.numpy()
            sample_images.append(img[0])

        sample_images = np.array(sample_images)

        if save_dir is None:
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)

    def discriminator(self):

        r"""Discriminator module for CycleGAN. Use it as a regular TensorFlow 2.0 Keras Model.

        Return:
            A tf.keras model  
        """

        kernel_initializer = self.config["kernel_initializer"]
        kernel_size = self.config["kernel_size"]
        disc_channels = self.config["disc_channels"]

        inputs = Input(shape=self.img_size)
        x = inputs

        down_stack = []
        for i, channel in enumerate(disc_channels[:-1]):
            if i == 0:
                down_stack.append(
                    self._downsample(
                        channel,
                        kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        batchnorm=False,
                    )
                )
            else:
                down_stack.append(
                    self._downsample(
                        channel, kernel_size=kernel_size, kernel_initializer=kernel_initializer
                    )
                )

        down_stack.append(ZeroPadding2D())
        down_stack.append(
            Conv2D(
                disc_channels[-1],
                kernel_size=kernel_size,
                strides=1,
                kernel_initializer=kernel_initializer,
                use_bias=False,
            )
        )

        down_stack.append(BatchNormalization())
        down_stack.append(LeakyReLU())
        down_stack.append(ZeroPadding2D())

        last = Conv2D(
            1, kernel_size=kernel_size, strides=1, kernel_initializer=kernel_initializer
        )

        for down in down_stack:
            x = down(x)

        out = last(x)
        model = Model(inputs=inputs, outputs=out)
        return model

    def __load_model(self):
        """
        Call build model to initialize the two generators and discriminators

        Note: Forward and backward GANs have the same architecture
        """

        self.gen_model_g, self.gen_model_f = self.generator(), self.generator()
        self.disc_model_x, self.disc_model_y = self.discriminator(), self.discriminator()

    def _save_samples(self, model, image, count):

        assert os.path.exists(self.save_img_dir), "sample directory does not exist"

        pred = model(image, training=False)
        pred = pred.numpy()
        image = image.numpy()

        curr_dir = os.path.join(self.save_img_dir, count)

        try:
            os.mkdir(curr_dir)
        except OSError:
            pass

        sample = 0
        for input_image, prediction in zip(image, pred):

            imageio.imwrite(
                os.path.join(curr_dir, "input_image_" + str(sample) + ".png"), input_image
            )

            imageio.imwrite(
                os.path.join(curr_dir, "translated_image_" + str(sample) + ".png"), prediction
            )
            sample += 1

    def fit(
        self,
        trainA=None,
        trainB=None,
        testA=None,
        testB=None,
        epochs=150,
        gen_g_optimizer="Adam",
        gen_f_optimizer="Adam",
        disc_x_optimizer="Adam",
        disc_y_optimizer="Adam",
        verbose=1,
        gen_g_learning_rate=2e-4,
        gen_f_learning_rate=2e-4,
        disc_x_learning_rate=2e-4,
        disc_y_learning_rate=2e-4,
        beta_1=0.5,
        tensorboard=False,
        save_model=None,
        LAMBDA=100,
        save_img_per_epoch=30,
    ):

        r"""Function to train the model

        Args:
            trainA (tf.data object): training data A
            trainB (tf.data object): training data B
            testA (tf.data object): testing data A
            testB (tf.data object): testing data B
            epochs (int, optional): number of epochs to train the model. Defaults to ``150``
            gen_g_optimizer (str, optional): optimizer used to train generator `G`. Defaults to ``Adam``
            gen_F_optimizer (str, optional): optimizer used to train generator `F`. Defaults to ``Adam``
            disc_x_optimizer (str, optional): optimizer used to train discriminator `X`. Defaults to ``Adam``
            disc_y_optimizer (str, optional): optimizer used to train discriminator `Y`. Defaults to ``Adam``
            verbose (int, optional): 1 - prints training outputs, 0 - no outputs. Defaults to ``1``
            gen_g_learning_rate (float, optional): learning rate of the generator `G` optimizer. Defaults to ``2e-4``
            gen_f_learning_rate (float, optional): learning rate of the generator `F` optimizer. Defaults to ``2e-4``
            disc_x_learning_rate (float, optional): learning rate of the discriminator `X` optimizer. Defaults to ``2e-4``
            disc_y_learning_rate (float, optional): learning rate of the discriminator `Y` optimizer. Defaults to ``2e-4``
            beta_1 (float, optional): decay rate of the first momement. set if ``Adam`` optimizer is used. Defaults to ``0.5``
            tensorboard (bool, optional): if true, writes loss values to ``logs/gradient_tape`` directory
                which aids visualization. Defaults to ``False``
            save_model (str, optional): Directory to save the trained model. Defaults to ``None``
            LAMBDA (int, optional): used to calculate generator loss. Defaults to ``100``
            save_img_per_epoch (int, optional): frequency of saving images during training. Defaults to ``30``
        """

        assert trainA is not None, "Initialize training data A through trainA parameter"
        assert trainB is not None, "Initialize training data B through trainB parameter"
        assert testA is not None, "Initialize testing data A through testA parameter"
        assert testB is not None, "Initialize testing data B through testB parameter"

        self.LAMBDA = LAMBDA

        self.__load_model()

        kwargs = {}
        kwargs["learning_rate"] = gen_g_learning_rate
        if gen_g_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        gen_g_optimizer = getattr(tf.keras.optimizers, gen_g_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = gen_f_learning_rate
        if gen_f_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        gen_f_optimizer = getattr(tf.keras.optimizers, gen_f_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = disc_x_learning_rate
        if disc_x_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        disc_x_optimizer = getattr(tf.keras.optimizers, disc_x_optimizer)(**kwargs)

        kwargs = {}
        kwargs["learning_rate"] = disc_y_learning_rate
        if disc_y_optimizer == "Adam":
            kwargs["beta_1"] = beta_1
        disc_y_optimizer = getattr(tf.keras.optimizers, disc_y_optimizer)(**kwargs)

        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/" + current_time + "/train"
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        curr_dir = os.getcwd()
        try:
            os.mkdir(os.path.join(curr_dir, "cyclegan_samples"))
        except OSError:
            pass

        self.save_img_dir = os.path.join(curr_dir, "cyclegan_samples")

        generator_g_loss = tf.keras.metrics.Mean()
        discriminator_x_loss = tf.keras.metrics.Mean()

        generator_f_loss = tf.keras.metrics.Mean()
        discriminator_y_loss = tf.keras.metrics.Mean()

        try:
            total = tf.data.experimental.cardinality(trainA).numpy()
        except:
            total = 0

        total = total if (total > 0) else len(list(trainA))

        for epoch in range(epochs):

            generator_g_loss.reset_states()
            generator_f_loss.reset_states()

            discriminator_x_loss.reset_states()
            discriminator_y_loss.reset_states()

            pbar = tqdm(total=total, desc="Epoch - " + str(epoch + 1))
            for image_x, image_y in tf.data.Dataset.zip((trainA, trainB)):

                with tf.GradientTape(persistent=True) as tape:

                    fake_y = self.gen_model_g(image_x, training=True)
                    cycled_x = self.gen_model_f(fake_y, training=True)

                    fake_x = self.gen_model_f(image_y, training=True)
                    cycled_y = self.gen_model_g(fake_x, training=True)

                    same_x = self.gen_model_f(image_x, training=True)
                    same_y = self.gen_model_g(image_y, training=True)

                    disc_real_x = self.disc_model_x(image_x, training=True)
                    disc_real_y = self.disc_model_y(image_y, training=True)

                    disc_fake_x = self.disc_model_x(fake_x, training=True)
                    disc_fake_y = self.disc_model_y(fake_y, training=True)

                    gen_g_loss = gan_generator_loss(disc_fake_y)
                    gen_f_loss = gan_generator_loss(disc_fake_x)

                    total_cycle_loss = cycle_loss(image_x, cycled_x, self.LAMBDA) + cycle_loss(
                        image_y, cycled_y, self.LAMBDA
                    )

                    total_gen_g_loss = (
                        gen_g_loss
                        + total_cycle_loss
                        + identity_loss(image_y, same_y, self.LAMBDA)
                    )
                    total_gen_f_loss = (
                        gen_f_loss
                        + total_cycle_loss
                        + identity_loss(image_x, same_x, self.LAMBDA)
                    )

                    disc_x_loss = gan_discriminator_loss(disc_real_x, disc_fake_x)
                    disc_y_loss = gan_discriminator_loss(disc_real_y, disc_fake_y)

                generator_g_gradients = tape.gradient(
                    total_gen_g_loss, self.gen_model_g.trainable_variables
                )
                generator_f_gradients = tape.gradient(
                    total_gen_f_loss, self.gen_model_f.trainable_variables
                )

                discriminator_x_gradients = tape.gradient(
                    disc_x_loss, self.disc_model_x.trainable_variables
                )
                discriminator_y_gradients = tape.gradient(
                    disc_y_loss, self.disc_model_y.trainable_variables
                )

                gen_g_optimizer.apply_gradients(
                    zip(generator_g_gradients, self.gen_model_g.trainable_variables)
                )
                gen_f_optimizer.apply_gradients(
                    zip(generator_f_gradients, self.gen_model_f.trainable_variables)
                )

                disc_x_optimizer.apply_gradients(
                    zip(discriminator_x_gradients, self.disc_model_x.trainable_variables)
                )
                disc_y_optimizer.apply_gradients(
                    zip(discriminator_y_gradients, self.disc_model_y.trainable_variables)
                )

                generator_g_loss(total_gen_g_loss)
                generator_f_loss(total_gen_f_loss)

                discriminator_x_loss(disc_x_loss)
                discriminator_y_loss(disc_y_loss)

                steps += 1
                pbar.update(1)

                if tensorboard:
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            "Generator_G_loss", total_gen_g_loss.numpy(), step=steps
                        )
                        tf.summary.scalar(
                            "Generator_F_loss", total_gen_f_loss.numpy(), step=steps
                        )
                        tf.summary.scalar(
                            "Discriminator_X_loss", disc_x_loss.numpy(), step=steps
                        )
                        tf.summary.scalar(
                            "Discriminator_Y_loss", disc_y_loss.numpy(), step=steps
                        )

            if epoch % save_img_per_epoch == 0:
                for image in testA.take(1):
                    self._save_samples(self.gen_model_g, image, str(epoch))

            if verbose == 1:
                print(
                    "Epoch:",
                    epoch + 1,
                    "Generator_G_loss:",
                    generator_g_loss.result().numpy(),
                    "Generator_F_loss:",
                    generator_f_loss.result().numpy(),
                    "Discriminator_X_loss:",
                    discriminator_x_loss.result().numpy(),
                    "Discriminator_Y_loss:",
                    discriminator_y_loss.result().numpy(),
                )

        if save_model is not None:

            assert isinstance(save_model, str), "Not a valid directory"
            if save_model[-1] != "/":
                self.gen_model_g.save_weights(save_model + "/generator_g_checkpoint")
            else:
                self.gen_model_g.save_weights(save_model + "generator_g_checkpoint")

    def generate_samples(self, test_ds=None, save_dir=None):

        r"""Generate samples using the trained model

        Args:
            test_ds (tf.data object): test data object used to generate samples`
            save_dir (str, optional): directory to save the generated images. Defaults to ``None``

        Return:
            returns ``None`` if save_dir is ``not None``, otherwise returns a numpy array with generated samples
        """

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = []
        for image in test_ds:
            gen_image = self.gen_model_g(image, training=False).numpy()
            generated_samples.append(gen_image[0])

        generated_samples = np.array(generated_samples)
        if save_dir is None:
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            imageio.imwrite(os.path.join(save_dir, "sample_" + str(i) + ".jpg"), sample)
