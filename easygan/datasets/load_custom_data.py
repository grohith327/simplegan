import tensorflow as tf
import numpy as np
import cv2
import glob
from tqdm import tqdm
import os 

'''
Dataset is loaded from tensorflow keras datasets

Function load_data returns a numpy array of shape (-1, 64, 64, 3) by default
'''


def load_custom_data(datadir=None, img_shape=(64, 64)):

    assert datadir is not None, "Enter a valid directory"
    assert len(img_shape) == 2 and isinstance(
        img_shape, tuple), "img_shape must be a tuple of size 2"

    train_data = []
    files = glob.glob(os.path.join(datadir, '*'))
    for file in tqdm(files, desc="Loading images"):
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, img_shape, interpolation=cv2.INTER_AREA)
            train_data.append(image)
        except BaseException:
            print("Error: Unable to load an image from directory")
            pass

    assert len(train_data) > 0, "No images to load from directory"

    train_data = np.array(train_data).astype('float32')

    return train_data


def load_custom_data_with_labels(datadir=None, img_shape=(64, 64)):

    assert datadir is not None, "Enter a valid directory"
    assert len(img_shape) == 2 and isinstance(
        img_shape, tuple), "img_shape must be a tuple of size 2"

    train_data = []
    labels = []
    files = glob.glob(os.path.join(datadir, '*/*'))

    for file in tqdm(files, desc="Loading images"):
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, img_shape, interpolation=cv2.INTER_AREA)
            train_data.append(image)
            label_name = int(file.split('/')[-2])
            labels.append(label_name)
        except BaseException:
            print("Error: Unable to load an image from directory")
            pass

    assert len(train_data) > 0, "No images to load from directory"

    train_data = np.array(train_data).astype('float32')
    labels = np.array(labels)
    labels = labels.reshape((-1, 1))

    return train_data, labels
