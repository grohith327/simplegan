import tensorflow as tf

'''
Computes a pixel wise mse loss - used in autoencoders
'''

__all__ = ['mse_loss']

def mse_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.pow((y_true - y_pred), 2))
