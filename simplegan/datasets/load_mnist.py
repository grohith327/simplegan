import tensorflow as tf
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_mnist returns a numpy array of shape (-1, 28, 28, 1)
'''

__all__ = ['load_mnist',
           'load_mnist_with_labels',
           'load_mnist_AE']


def load_mnist():

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_data = x_train.astype('float32')

    train_data = train_data.reshape((-1, 28, 28, 1))

    return train_data


def load_mnist_AE():

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    train_data = x_train.astype('float32')
    test_data = x_test.astype('float32')

    train_data = train_data.reshape((-1, 28, 28, 1))
    test_data = test_data.reshape((-1, 28, 28, 1))

    return train_data, test_data


def load_mnist_with_labels():

    (x_train, train_labels), (x_test,test_labels) = tf.keras.datasets.mnist.load_data()

    train_data = np.concatenate((x_train, x_test), 0)
    train_labels = np.concatenate((train_labels, test_labels), 0)

    train_data = train_data.astype('float32')
    
    train_data = train_data.reshape((-1, 28, 28, 1))
    train_labels = train_labels.reshape((-1, 1))

    return train_data, train_labels
