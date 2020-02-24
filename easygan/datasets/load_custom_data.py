import tensorflow as tf
import numpy as np
import cv2
import glob
from tqdm import tqdm

'''
Dataset is loaded from tensorflow keras datasets

Function load_data returns a numpy array of shape (-1, 64, 64, 3) by default
'''

def load_data(datadir = None, img_shape = (64, 64)):

	assert datadir is not None, "Enter a valid directory"
	assert len(t) == 2 and type(img_shape) == tuple, "img_shape must be a tuple of size 2"

	train_data = []
	files = glob.glob(datadir)
	for file in tqdm(files, desc="Loading images"):
		try:
			image = cv2.imread(file)
			image = cv2.resize(image, img_shape, interpolation = cv2.INTER_AREA)
			train_data.append(image)
		except:
			print("Error: Unable to load an image from directory")
			pass

	assert len(train_data) > 0, "No images to load from directory"

	train_data = np.array(train_data).astype('float32')

	return train_data	



