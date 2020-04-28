import tensorflow as tf

__all__ = ["hinge_loss_generator", "hinge_loss_discriminator"]


def hinge_loss_generator(generated_output):
    r"""
    Args:
        generated_output (tensor): A tensor of the generated image.

    Return:
        a tensor representing hinge loss.
    """
    return -tf.reduce_mean(generated_output)


def hinge_loss_discriminator(real_output, generated_output):
    r"""
    Args:
        real_output (tensor): A tensor of real output.
        generated_output (tensor): A tensor of predictions made by discriminator.

    Return:
        a tensor representing hinge loss.
    """
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
    generated_loss = tf.reduce_mean(tf.nn.relu(1 + generated_output))
    return real_loss + generated_loss
