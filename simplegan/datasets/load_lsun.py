import tensorflow as tf
import cv2
from tqdm import tqdm
import tensorflow_datasets as tfds
import numpy as np

'''
Dataset is loaded from tensorflow_datasets

Function load_lsun returns a numpy array of shape (-1, 64, 64, 3)
'''

__all__ = ['load_lsun']


def load_lsun(info=False, img_shape=(64, 64)):

    assert len(img_shape) == 2 and isinstance(img_shape, tuple), "img_shape must be a tuple of size 2"

    if(info):

        ds_train, info = tfds.load(name="lsun", 
                                split='train', 
                                shuffle_files=True, 
                                with_info=info)
    else:
        ds_train = tfds.load(name="lsun",
                            split='train',
                            shuffle_files=True,
                            with_info=info)

    train_data = []
    with tqdm(total=100, desc='preparing dataset') as pbar:
        for i, data in enumerate(ds_train):

            img = data['image'].numpy()
            img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
            train_data.append(img)

            if(i % 1681 == 0):
                pbar.update(1)

    train_data = np.array(train_data).astype('float32')

    if(info):
        return train_data, info

    return train_data
