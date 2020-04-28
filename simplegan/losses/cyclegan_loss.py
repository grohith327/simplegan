import tensorflow as tf

__all__ = ["cycle_loss", "identity_loss"]


def cycle_loss(real_img, cycle_img, LAMBDA):

    r"""
    Args:
        real_img (tensor): A tensor representing the real image
        cycle_img (tensor): A tensor representing the generated image
        LAMBDA (int): An integer to scale the loss

    Return:
        a tensor representing the loss
    """

    loss = tf.math.reduce_mean(tf.math.abs(real_img - cycle_img))
    loss *= LAMBDA
    return loss


def identity_loss(real_img, same_img, LAMBDA):

    r"""
    Args:
        real_img (tensor): A tensor representing the real image
        cycle_img (tensor): A tensor representing the generated image
        LAMBDA (int): An integer to scale the loss
    
    Return:
        a tensor representing the loss
    """

    loss = tf.reduce_mean(tf.math.abs(real_img - same_img))
    loss *= LAMBDA
    return loss * 0.5
