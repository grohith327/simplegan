import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose, Dense, Reshape, Flatten
from tensorflow.keras import Model
from datasets.load_cifar10 import load_cifar10
from datasets.load_mnist import load_mnist
from datasets.load_custom_data import load_custom_data
from datasets.load_cifar100 import load_cifar100
from gan.dcgan import DCGAN
from losses.wasserstein_loss import wgan_discriminator_loss, wgan_generator_loss
import cv2
import numpy as np
import datetime
from datasets.load_lsun import load_lsun
import sys
sys.path.append('..')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WGAN(DCGAN):

    def __init__(self):
        DCGAN.__init__(self)

    def fit(self,
            train_ds=None,
            epochs=100,
            gen_optimizer='RMSprop',
            disc_optimizer='RMSprop',
            print_steps=100,
            gen_learning_rate=5e-5,
            disc_learning_rate=5e-5,
            beta_1=0.5,
            tensorboard=False,
            save_model=None):

        assert train_ds is not None, 'Initialize training data through train_ds parameter'

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

        for epoch in range(epochs):
            for data in train_ds:

                for _ in range(5):
                    with tf.GradientTape() as tape:

                        Z = tf.random.normal([data.shape[0], self.noise_dim])
                        fake = self.gen_model(Z)
                        fake_logits = self.disc_model(fake)
                        real_logits = self.disc_model(data)
                        D_loss = wgan_discriminator_loss(
                            real_logits, fake_logits)

                    gradients = tape.gradient(
                        D_loss, self.disc_model.trainable_variables)
                    clipped_gradients = [
                        (tf.clip_by_value(grad, -0.01, 0.01)) for grad in gradients]
                    disc_optimizer.apply_gradients(
                        zip(clipped_gradients, self.disc_model.trainable_variables))

                with tf.GradientTape() as tape:

                    Z = tf.random.normal([data.shape[0], self.noise_dim])
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = wgan_generator_loss(fake_logits)

                gradients = tape.gradient(
                    G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(
                    zip(gradients, self.gen_model.trainable_variables))

                if(steps % print_steps == 0):
                    print(
                        'Step:',
                        steps + 1,
                        'D_loss:',
                        D_loss.numpy(),
                        'G_loss',
                        G_loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'discr_loss', D_loss.numpy(), step=steps)
                        tf.summary.scalar(
                            'genr_loss', G_loss.numpy(), step=steps)

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
