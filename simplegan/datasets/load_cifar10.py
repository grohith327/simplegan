import tensorflow as tf
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_cifar10 returns a numpy array of shape (-1, 32, 32, 3)
'''

__all__ = ['load_cifar10',
           'load_cifar10_with_labels',
           'load_cifar10_AE']


def load_cifar10():

    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_data = x_train.astype('float32')

    return train_data


def load_cifar10_AE():

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    train_data = x_train.astype('float32')
    test_data = x_test.astype('float32')

    return train_data, test_data


def load_cifar10_with_labels():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_data = np.concatenate((x_train, x_test), 0)
    train_labels = np.concatenate((y_train, y_test), 0)

    train_data = train_data.astype('float32')

    return train_data, train_labels
