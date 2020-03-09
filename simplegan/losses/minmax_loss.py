import tensorflow as tf

'''
Returns a generator and discriminator loss.

Insipired by the Original GAN paper: Goodfellow et al.
'''

__all__ = ['gan_discriminator_loss',
           'gan_generator_loss']

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def gan_discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def gan_generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
