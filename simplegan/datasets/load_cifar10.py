import tensorflow as tf
import numpy as np

__all__ = ["load_cifar10", "load_cifar10_with_labels", "load_cifar10_AE"]


def load_cifar10():
    r"""Loads the Cifar10 training data without labels - used in GANs

    Args:
        None

    Return:
        a numpy array of shape (-1, 32, 32, 3)

    """

    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_data = x_train.astype("float32")

    return train_data


def load_cifar10_AE():
    r"""Loads the Cifar10 training and testing data without labels - used in Autoencoder

    Args:
        None

    Return:
        two numpy arrays of shape (-1, 32, 32, 3) each

    """

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    train_data = x_train.astype("float32")
    test_data = x_test.astype("float32")

    return train_data, test_data


def load_cifar10_with_labels():
    r"""Loads the Cifar10 train and test data along with labels and concatenates them - used in CGAN

    Args:
        None

    Return:
        two numpy arrays, one of shape (-1, 32, 32, 3) which represents features and the other of shape (-1, 1) which represents labels
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_images = np.concatenate((x_train, x_test), 0)
    train_labels = np.concatenate((y_train, y_test), 0)

    train_images = train_images.astype("float32")
    train_labels = train_labels.astype("int32")
    return train_images, train_labels
