import tensorflow as tf

'''
loss functions used in pix2pix model
'''

__all__ = ['pix2pix_generator_loss',
           'pix2pix_discriminator_loss']

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def pix2pix_generator_loss(disc_fake_output, fake, real, l):

    gan_loss = cross_entropy(tf.ones_like(disc_fake_output), disc_fake_output)

    l1_loss = tf.math.reduce_mean(tf.math.abs(real - fake))

    total_gen_loss = gan_loss + l * l1_loss

    return total_gen_loss, gan_loss, l1_loss


def pix2pix_discriminator_loss(disc_real_output, disc_fake_output):

    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = cross_entropy(tf.ones_like(disc_fake_output),disc_fake_output)

    total_loss = real_loss + generated_loss

    return total_loss
