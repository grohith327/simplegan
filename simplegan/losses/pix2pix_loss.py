import tensorflow as tf

"""
loss functions used in pix2pix model
"""

__all__ = ["pix2pix_generator_loss", "pix2pix_discriminator_loss"]

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def pix2pix_generator_loss(disc_fake_output, fake, real, l):

    r"""
    Args:
        disc_fake_output (tensor): A tensor representing the fake logits of discriminator
        fake (tensor): A tensor representing the values from the generator
        real (tensor): A tensor representing the real values
        l (int): An integer to scale the l1 loss

    Return:
        total loss of generator, total loss of GAN and L1 loss
    """

    gan_loss = cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)

    l1_loss = tf.math.reduce_mean(tf.math.abs(real - fake))

    total_gen_loss = gan_loss + l * l1_loss

    return total_gen_loss, gan_loss, l1_loss


def pix2pix_discriminator_loss(disc_real_output, disc_fake_output):

    r"""
    Args:
        disc_real_output (tensor): A tensor representing the real logits of the discriminator
        disc_fake_output (tensor): A tensor representing the fake logits of the discriminator

    Return:
        total loss of discriminator
    """

    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)

    total_loss = real_loss + generated_loss

    return total_loss
