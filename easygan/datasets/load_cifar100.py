import tensorflow as tf
import numpy as np

'''
Dataset is loaded from tensorflow keras datasets

Function load_cifar100 returns a numpy array of shape (-1, 32, 32, 3)
'''

def load_cifar100():

	(x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()

	train_data = np.concatenate((x_train, x_test), 0)

	train_data = train_data.astype('float32')

	return train_data
