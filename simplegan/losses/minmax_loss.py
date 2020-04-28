import tensorflow as tf


__all__ = ["gan_discriminator_loss", "gan_generator_loss"]

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def gan_discriminator_loss(real_output, fake_output):

    r"""
    Args:
        real_output (tensor): A tensor representing the real logits of discriminator
        fake_output (tensor): A tensor representing the fake logits of discriminator

    Return:
        a tensor representing the sum of real and fake loss
    """

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def gan_generator_loss(fake_output):

    r"""
    Args:
        fake_output (tensor): A tensor representing the fake logits of discriminator

    Return:
        a tensor representing the generator loss
    """

    return cross_entropy(tf.ones_like(fake_output), fake_output)
