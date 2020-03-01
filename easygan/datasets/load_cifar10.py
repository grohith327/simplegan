import tensorflow as tf
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_cifar10 returns a numpy array of shape (-1, 32, 32, 3)
'''

def load_cifar10():

    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

    train_data = x_train.astype('float32')

    return train_data