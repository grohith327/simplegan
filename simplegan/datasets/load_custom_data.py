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

__all__ = ['load_custom_data',
           'load_custom_data_AE',
           'load_custom_data_with_labels']


def load_custom_data(datadir=None, img_shape=(64, 64)):

    error_message = "Enter a valid directory \n Directory structure: \n {} \n {} -*jpg".format(
        datadir, ' ' * 2)
    assert datadir is not None, error_message
    assert len(img_shape) == 2 and isinstance(img_shape, tuple), "img_shape must be a tuple of size 2"

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


def load_custom_data_AE(datadir=None, img_shape=(64, 64)):

    assert datadir is not None, "Enter a valid directory"

    error_message = "train directory not found \n Directory structure: \n {} \n {} -train \n {} -*.jpg \n {} -test \n {} -*.jpg".format(
        datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
    assert os.path.exists(os.path.join(datadir, 'train')), error_message

    error_message = "test directory not found \n Directory structure: \n {} \n {} -train \n {} -*.jpg \n {} -test \n {} -*.jpg".format(
        datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
    assert os.path.exists(os.path.join(datadir, 'test')), error_message

    assert len(img_shape) == 2 and isinstance(img_shape, tuple), "img_shape must be a tuple of size 2"

    train_data = []

    files = glob.glob(os.path.join(datadir, 'train/*'))
    for file in tqdm(files, desc="Loading train images"):
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, img_shape, interpolation=cv2.INTER_AREA)
            train_data.append(image)
        except BaseException:
            print("Error: Unable to load an image from directory")
            pass

    assert len(train_data) > 0, "No images to load from train directory"

    test_data = []

    files = glob.glob(os.path.join(datadir, 'test/*'))
    for file in tqdm(files, desc="Loading test images"):
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, img_shape, interpolation=cv2.INTER_AREA)
            test_data.append(image)
        except BaseException:
            print("Error: Unable to load an image from directory")
            pass

    assert len(test_data) > 0, "No images to load from test directory"

    train_data = np.array(train_data).astype('float32')
    test_data = np.array(test_data).astype('float32')

    return train_data, test_data


def load_custom_data_with_labels(datadir=None, img_shape=(64, 64)):

    assert datadir is not None, "Enter a valid directory"
    assert len(img_shape) == 2 and isinstance(img_shape, tuple), "img_shape must be a tuple of size 2"

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
        except ValueError:
            print("Ensure Directory is of following structure: \n {} \n {} -label 1(int type) \n {} -*.jpg \n {} -label 2(int type) \n {} -*.jpg \n {} ...".format(
                datadir, ' '*2, ' '*4, ' '*2, ' '*4, ' '*2))
            pass

    assert len(train_data) > 0, "No images to load from directory"

    train_data = np.array(train_data).astype('float32')
    labels = np.array(labels)
    labels = labels.reshape((-1, 1))

    return train_data, labels
