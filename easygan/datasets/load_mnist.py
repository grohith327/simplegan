import tensorflow as tf 
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_mnist returns a numpy array of shape (-1, 28, 28, 1)
'''

def load_mnist():

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    train_data = x_train.astype('float32')

    train_data = train_data.reshape((-1, 28, 28, 1))

    return train_data