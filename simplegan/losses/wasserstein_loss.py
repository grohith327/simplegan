import tensorflow as tf

__all__ = ["wgan_discriminator_loss", "wgan_generator_loss"]


def wgan_discriminator_loss(real_output, fake_output):

    r"""
    Args:
        real_output (tensor): a tensor representing the real logits of the discriminator
        fake_output (tensor): a tensor representing the fake logits of the discriminator

    Return:
        total discriminator loss
    """

    total_loss = tf.math.reduce_mean(real_output) - tf.math.reduce_mean(fake_output)
    return total_loss


def wgan_generator_loss(fake_output):

    r"""
    Args:
        fake_output (tensor): a tensor representing the fake logits of the discriminator

    Return:
        generator loss
    """

    return -tf.math.reduce_mean(fake_output)
