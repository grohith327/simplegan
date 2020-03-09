import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

'''
returns train_dataA, train_dataB, test_dataA, test_dataB -> tensorflow dataset


paper: https://arxiv.org/abs/1703.10593

Code inspired from: https://www.tensorflow.org/tutorials/generative/cyclegan#import_and_reuse_the_pix2pix_models
'''

__all__ = ['cyclegan_dataloader']

class cyclegan_dataloader:

    def __init__(self, dataset_name=None, img_width=256, img_height=256,
                 datadir=None):

        self.dataset_name = dataset_name
        self.img_width = img_width
        self.img_height = img_height
        self.datadir = datadir
        self.channels = 3

    def _random_crop(self, image):

        cropped_image = tf.image.random_crop(
            image,
            size=[
                self.img_height,
                self.img_width,
                self.channels])
        return cropped_image

    def _normalize(self, image):

        img = tf.cast(image, tf.float32)
        img = (img / 127.5) - 1
        return img

    def _resize(self, image, height, width):

        image = tf.image.resize(
            image, [
                height, 
                width], 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def _random_jitter(self, image):

        image = self._resize(image, 286, 286)
        image = self._random_crop(image)
        image = tf.image.random_flip_left_right(image)

        return image

    def _preprocess_image_train(self, image, label):

        image = self._random_jitter(image)
        image = self._normalize(image)
        return image

    def _preprocess_image_test(self, image, label):

        image = self._normalize(image)
        image = self._resize(image, self.img_height, self.img_width)
        return image

    def _load_cyclegan_data(self):

        dataset_list = [
            'apple2orange',
            'summer2winter_yosemite',
            'horse2zebra',
            'monet2photo',
            'cezanne2photo',
            'ukiyoe2photo',
            'vangogh2photo',
            'maps',
            'cityscapes',
            'facades',
            'iphone2dslr_flower']

        assert self.dataset_name in dataset_list, "Dataset name is not a valid member of " + \
            ','.join(dataset_list)

        load_data = os.path.join('cycle_gan', self.dataset_name)

        dataset = tfds.load(load_data, as_supervised=True)
        trainA, trainB = dataset['trainA'], dataset['trainB']
        testA, testB = dataset['testA'], dataset['testB']

        trainA = trainA.map(self._preprocess_image_train)
        trainB = trainB.map(self._preprocess_image_train)

        testA = testA.map(self._preprocess_image_test)
        testB = testB.map(self._preprocess_image_test)

        self.channels = 3

        return trainA, trainB, testA, testB

    def _load__train_image(self, filename):

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)

        image = tf.cast(image, tf.float32)

        image = self._random_jitter(image)
        image = self._normalize(image)

        return image

    def _load__test_image(self, filename):

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)

        image = tf.cast(image, tf.float32)

        image = self._resize(image, self.img_height, self.img_width)
        image = self._normalize(image)

        return image

    def _load_custom_data(self):

        error_message = "trainA directory not found \n Directory structure: \n {} \n {} -trainA \n {} -*.jpg \n {} -trainB \n {} -*.jpg \n {} -testA \n {} -*.jpg \n {} -testB \n {} -*.jpg".format(
            self.datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
        assert os.path.exists(os.path.join(self.datadir, 'trainA')), error_message

        train_data = tf.data.Dataset.list_files(
            os.path.join(self.datadir, 'trainA/*.jpg'))
        trainA = train_data.map(self._load__train_image)

        error_message = "trainB directory not found \n Directory structure: \n {} \n {} -trainA \n {} -*.jpg \n {} -trainB \n {} -*.jpg \n {} -testA \n {} -*.jpg \n {} -testB \n {} -*.jpg".format(
            self.datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
        assert os.path.exists(os.path.join(self.datadir, 'trainB')), error_message

        train_data = tf.data.Dataset.list_files(
            os.path.join(self.datadir, 'trainB/*.jpg'))
        trainB = train_data.map(self._load__train_image)

        error_message = "testA directory not found \n Directory structure: \n {} \n {} -trainA \n {} -*.jpg \n {} -trainB \n {} -*.jpg \n {} -testA \n {} -*.jpg \n {} -testB \n {} -*.jpg".format(
            self.datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
        assert os.path.exists(os.path.join(self.datadir, 'testA')), error_message

        test_data = tf.data.Dataset.list_files(
            os.path.join(self.datadir, 'testA/*.jpg'))
        testA = test_data.map(self._load__test_image)

        error_message = "testB directory not found \n Directory structure: \n {} \n {} -trainA \n {} -*.jpg \n {} -trainB \n {} -*.jpg \n {} -testA \n {} -*.jpg \n {} -testB \n {} -*.jpg".format(
            self.datadir, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4, ' ' * 2, ' ' * 4)
        assert os.path.exists(os.path.join(self.datadir, 'testB')), error_message

        test_data = tf.data.Dataset.list_files(
            os.path.join(self.datadir, 'testB/*.jpg'))
        testB = test_data.map(self._load__test_image)

        return trainA, trainB, testA, testB

    def load_dataset(self):

        assert self.dataset_name is not None or self.datadir is not None, "Enter directory to load custom data or choose from exisisting data to load from"

        if(self.dataset_name is not None):

            trainA, trainB, testA, testB = self._load_cyclegan_data()

        else:
            
            trainA, trainB, testA, testB = self._load_custom_data()

        return trainA, trainB, testA, testB
