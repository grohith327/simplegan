import tensorflow as tf
import numpy as np

__all__ = ["load_mnist", "load_mnist_with_labels", "load_mnist_AE"]


def load_mnist():

    r"""Loads the MNIST training data without labels - used in GANs

    Args:
        None

    Return:
        a numpy array of shape (-1, 28, 28, 1)

    """

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_data = x_train.astype("float32")

    train_data = train_data.reshape((-1, 28, 28, 1))

    return train_data


def load_mnist_AE():

    r"""Loads the MNIST training and testing data without labels - used in Autoencoder

    Args:
        None

    Return:
        two numpy arrays of shape (-1, 28, 28, 1) each

    """

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    train_data = x_train.astype("float32")
    test_data = x_test.astype("float32")

    train_data = train_data.reshape((-1, 28, 28, 1))
    test_data = test_data.reshape((-1, 28, 28, 1))

    return train_data, test_data


def load_mnist_with_labels():

    r"""Loads the MNIST train and test data along with labels and concatenates them - used in CGAN

    Args:
        None

    Return:
        two numpy arrays, one of shape (-1, 28, 28, 1) which represents features and the other of shape (-1, 1) which represents labels
    """

    (x_train, train_labels), (x_test, test_labels) = tf.keras.datasets.mnist.load_data()

    train_data = np.concatenate((x_train, x_test), 0)
    train_labels = np.concatenate((train_labels, test_labels), 0)

    train_data = train_data.astype("float32")

    train_data = train_data.reshape((-1, 28, 28, 1))
    train_labels = train_labels.reshape((-1, 1))

    return train_data, train_labels
