import tensorflow as tf
import numpy as np

"""
Dataset is loaded from tensorflow keras datasets

Function load_cifar100 returns a numpy array of shape (-1, 32, 32, 3)
"""

__all__ = ["load_cifar100"]


def load_cifar100():

    r"""Loads the Cifar100 training data without labels - used in GANs

    Args:
        None

    Return:
        a numpy array of shape (-1, 32, 32, 3)

    """

    (x_train, _), (_, _) = tf.keras.datasets.cifar100.load_data()

    train_data = x_train.astype("float32")

    return train_data
