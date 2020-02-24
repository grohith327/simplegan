import tensorflow as tf 
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_mnist returns a numpy array of shape (-1, 28, 28, 1)
'''

def load_mnist():

	(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

	train_data = np.concatenate((x_train, x_test), 0)

	train_data = train_data.astype('float32')

	train_data = train_data.reshape((-1, 28, 28, 1))

	return train_data
