import tensorflow as tf

'''
Returns a generator and discriminator loss.

WGAN paper: https://arxiv.org/abs/1701.07875
'''


def wgan_discriminator_loss(real_output, fake_output):
    total_loss = tf.math.reduce_mean(
        real_output) - tf.math.reduce_mean(fake_output)
    return total_loss


def wgan_generator_loss(fake_output):
    return -tf.math.reduce_mean(fake_output)
