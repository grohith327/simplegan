import os
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, ReLU, Input
from tensorflow.keras import Model
from ..datasets.load_pix2pix_datasets import pix2pix_dataloader
from ..losses.pix2pix_loss import pix2pix_generator_loss, pix2pix_discriminator_loss
import imageio
import cv2
import tensorflow as tf
import numpy as np
import datetime
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
Paper: https://arxiv.org/abs/1611.07004

The following code is inspired from: https://www.tensorflow.org/tutorials/generative/pix2pix#load_the_dataset

During trainig, samples will be saved at ./samples and saved rate at a rate given by save_img_per_epoch
'''


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning

__all__ = ['Pix2Pix']


class Pix2Pix:

    def __init__(self,
                kernel_initializer = tf.random_normal_initializer(0., 0.02),
                dropout_rate = 0.5,
                kernel_size = (
                    4,
                    4),
                gen_enc_channels = [
                    128,
                    256,
                    512,
                    512,
                    512,
                    512,
                    512],
                gen_dec_channels = [
                    512,
                    512,
                    512,
                    512,
                    256,
                    128,
                    64],
                disc_channels = [
                    64,
                    128,
                    256,
                    512]):

        self.gen_model = None
        self.disc_model = None
        self.channels = None
        self.LAMBDA = None
        self.img_size = None
        self.save_img_dir = None
        self.config = locals()

        assert len(self.config['gen_enc_channels']) == len(self.config['gen_dec_channels']), "Dimension mismatch: length of gen_enc_channels should match length of gen_dec_channels"
        test = self.config['gen_enc_channels'][:-1]
        test.reverse()
        assert test == self.config['gen_dec_channels'][:-1], "Number of channels in Enocder of generator should be equal to reverse of number of Decoder channels of generator"

    def load_data(self, 
                data_dir=None, 
                use_cityscapes=False,
                use_edges2handbags=False, 
                use_edges2shoes=False,
                use_facades=False, 
                use_maps=False, 
                batch_size=32):

        if(use_cityscapes):

            data_obj = pix2pix_dataloader(dataset_name='cityscapes')

        elif(use_edges2handbags):

            data_obj = pix2pix_dataloader(dataset_name='edges2handbags')

        elif(use_edges2shoes):

            data_obj = pix2pix_dataloader(dataset_name='edges2shoes')

        elif(use_facades):

            data_obj = pix2pix_dataloader(dataset_name='facades')

        elif(use_maps):

            data_obj = pix2pix_dataloader(dataset_name='maps')

        else:

            data_obj = pix2pix_dataloader(datadir=data_dir)

        train_ds, test_ds = data_obj.load_dataset()

        for data in train_ds.take(1):
            self.img_size = data[0].shape
            self.channels = data[0].shape[-1]

        train_ds = train_ds.shuffle(100000).batch(batch_size)
        test_ds = test_ds.shuffle(100000).batch(batch_size)

        return train_ds, test_ds

    def get_sample(self, data=None, n_samples=1, save_dir=None):

        assert data is not None, "Data not provided"

        sample_images = []
        data.unbatch()
        for input_img, target_img in data.take(n_samples):

            input_img = input_img.numpy()
            target_img = target_img.numpy()
            sample_images.append([input_img, target_img])

        sample_images = np.array(sample_images)

        if(save_dir is None):
            return sample_images

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(sample_images):

            input_img = sample[0]
            target_img = sample[1]

            imageio.imwrite(
                os.path.join(
                    save_dir,
                    'input_sample_' +
                    str(i) +
                    '.jpg'),
                input_img)

            imageio.imwrite(
                os.path.join(
                    save_dir,
                    'target_sample_' +
                    str(i) +
                    '.jpg'),
                target_img)

    def _downsample(self, filters, kernel_size, kernel_initializer,
                    batchnorm=True):

        model = tf.keras.Sequential()
        model.add(
            Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=2,
                kernel_initializer=kernel_initializer,
                padding='same',
                use_bias=False))

        if(batchnorm):
            model.add(BatchNormalization())

        model.add(LeakyReLU())

        return model

    def _upsample(self, filters, kernel_size, kernel_initializer,
                  dropout_rate=None, dropout=False):

        model = tf.keras.Sequential()
        model.add(
            Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False))
        model.add(BatchNormalization())

        if(dropout):
            model.add(Dropout(dropout_rate))

        model.add(ReLU())

        return model

    def generator(self, config):

        kernel_initializer = config['kernel_initializer']
        dropout_rate = config['dropout_rate']
        kernel_size = config['kernel_size']
        gen_enc_channels = config['gen_enc_channels']
        gen_dec_channels = config['gen_dec_channels']
        gen_enc_layers = len(gen_enc_channels)
        gen_dec_layers = len(gen_dec_channels)
        
        inputs = Input(shape=self.img_size)

        down_stack = [
            self._downsample(
                (gen_enc_channels[0] // 2),
                4,
                kernel_initializer,
                batchnorm=False)]

        for channel in gen_enc_channels:
            down_stack.append(
                self._downsample(
                    channel,
                    kernel_size,
                    kernel_initializer=kernel_initializer))

        up_stack = []
        for i, channel in enumerate(gen_dec_channels):
            if(i < 3):
                up_stack.append(
                    self._upsample(
                        channel,
                        kernel_size,
                        kernel_initializer=kernel_initializer,
                        dropout_rate=dropout_rate,
                        dropout=True))
            else:
                up_stack.append(
                    self._upsample(
                        channel,
                        kernel_size,
                        kernel_initializer=kernel_initializer))

        last = Conv2DTranspose(
            self.channels,
            strides=2,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            activation='tanh')

        x = inputs

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)

        model = Model(inputs=inputs, outputs=x)
        return model

    def discriminator(self, config):

        kernel_initializer = config['kernel_initializer']
        kernel_size = config['kernel_size']
        disc_channels = config['disc_channels']
        disc_layers = len(disc_channels)

        inputs = Input(shape=self.img_size)
        target = Input(shape=self.img_size)

        x = Concatenate()([inputs, target])

        down_stack = []
        for i, channel in enumerate(disc_channels[:-1]):
            if(i == 0):
                down_stack.append(
                    self._downsample(
                        channel,
                        kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        batchnorm=False))
            else:
                down_stack.append(
                    self._downsample(
                        channel,
                        kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer))

        down_stack.append(ZeroPadding2D())
        down_stack.append(Conv2D(disc_channels[-1],
                                 kernel_size=kernel_size,
                                 strides=1,
                                 kernel_initializer=kernel_initializer,
                                 use_bias=False))

        down_stack.append(BatchNormalization())
        down_stack.append(LeakyReLU())
        down_stack.append(ZeroPadding2D())

        last = Conv2D(1, 
                    kernel_size=kernel_size, 
                    strides=1,
                    kernel_initializer=kernel_initializer)

        for down in down_stack:
            x = down(x)

        out = last(x)
        model = Model(inputs=[inputs, target], outputs=out)
        return model

    def __load_model(self):

        self.gen_model, self.disc_model = self.generator(
            self.config), self.discriminator(self.config)

    def _save_samples(self, model, ex_input, ex_target, count):

        assert os.path.exists(
            self.save_img_dir), "sample directory does not exist"

        prediction = model(ex_input, training=False)

        input_images = ex_input.numpy()
        target_images = ex_target.numpy()
        predictions = prediction.numpy()

        curr_dir = os.path.join(self.save_img_dir, count)
        try:
            os.mkdir(curr_dir)
        except OSError:
            pass

        sample = 0
        for input_image, target_image, prediction in zip(
                input_images, target_images, predictions):

            imageio.imwrite(
                os.path.join(
                    curr_dir,
                    'input_image_' +
                    str(sample) +
                    '.png'),
                input_image)

            imageio.imwrite(
                os.path.join(
                    curr_dir,
                    'target_image_' +
                    str(sample) +
                    '.png'),
                target_image)

            imageio.imwrite(
                os.path.join(
                    curr_dir,
                    'translated_image_' +
                    str(sample) +
                    '.png'),
                prediction)
            sample += 1

    def fit(self,
            train_ds=None,
            test_ds=None,
            epochs=150,
            gen_optimizer='Adam',
            disc_optimizer='Adam',
            verbose=1,
            gen_learning_rate=2e-4,
            disc_learning_rate=2e-4,
            beta_1=0.5,
            tensorboard=False,
            save_model=None,
            LAMBDA=100,
            save_img_per_epoch=30):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'
        assert test_ds is not None, 'Initialize testing data through test_ds parameter'
        self.LAMBDA = LAMBDA

        self.__load_model()

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

        curr_dir = os.getcwd()
        try:
            os.mkdir(os.path.join(curr_dir, 'pix2pix_samples'))
        except OSError:
            pass
        self.save_img_dir = os.path.join(curr_dir, 'pix2pix_samples')

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
            for input_image, target_image in train_ds:

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    gen_output = self.gen_model(input_image, training=True)

                    disc_real_output = self.disc_model(
                        [input_image, target_image], training=True)
                    disc_gen_output = self.disc_model(
                        [input_image, gen_output], training=True)

                    gen_total_loss, gan_loss, l1_loss = pix2pix_generator_loss(
                        disc_gen_output, gen_output, target_image, self.LAMBDA)
                    disc_loss = pix2pix_discriminator_loss(
                        disc_real_output, disc_gen_output)

                gen_gradients = gen_tape.gradient(
                    gen_total_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gen_gradients, self.gen_model.trainable_variables))

                disc_gradients = disc_tape.gradient(
                    disc_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(
                    zip(disc_gradients, self.disc_model.trainable_variables))

                generator_loss(gen_total_loss)
                discriminator_loss(disc_loss)

                steps += 1
                pbar.update(1)

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'discr_loss', disc_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'total_gen_loss', gen_total_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'gan_loss', gan_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'l1_loss', l1_loss.numpy(), step=steps)

            if(epoch % save_img_per_epoch == 0):
                for input_image, target_image in test_ds.take(1):
                    self._save_samples(
                        self.gen_model, input_image, target_image, str(epoch))

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

    def generate_samples(self, test_ds=None, save_dir=None):

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = []
        for image in test_ds:
            gen_image = self.gen_model(image, training=False).numpy()
            generated_samples.append(gen_image[0])

        generated_samples = np.array(generated_samples)
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
