import tensorflow as tf

__all__ = ["mse_loss"]


def mse_loss(y_true, y_pred):

    r"""
    Args:
        y_true (tensor): A tensor representing the orginal values
        y_pred (tensor): A tensor representing the predicted values by the network

    Return:
        A tensor representing the pixel wise loss between orginal and predicted values
    """

    return tf.math.reduce_mean(tf.math.pow((y_true - y_pred), 2))
