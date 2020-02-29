import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os 

'''
returns train_dataA, train_dataB, test_dataA, test_dataB -> tensorflow dataset


paper: https://arxiv.org/abs/1703.10593

Code inspired from: https://www.tensorflow.org/tutorials/generative/cyclegan#import_and_reuse_the_pix2pix_models
'''

class cyclegan_dataloader:

    def __init__(self, dataset_name = None, img_width = 256, img_height = 256, 
                datadir = None):

        self.dataset_name = dataset_name
        self.img_width = img_width
        self.img_height = img_height
        self.datadir = datadir
        self.channels = None


    def __random_crop(self, image):

        cropped_image = tf.image.random_crop(image, size = [self.img_height,
                                            self.img_width, self.channels])
        return cropped_image

    def _normalize(self, image):

        img = tf.cast(image, tf.float32)
        img = (img / 127.5) - 1
        return img

    def _random_jitter(self, image):

        image = tf.image.resize(image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = self._random_crop(image)
        image = tf.image.random_flip_left_right(image)

        return image

    def _preprocess_image_train(self, image, label):

        image = self._random_jitter(image)
        image = self._normalize(image)
        return image

    def _preprocess_image_test(self, image, label):

        image = self._normalize(image)
        return image

    def _load_cyclegan_data(self):

        dataset_list = ['apple2orange', 'summer2winter_yosemite', 'horse2zebra',
                        'monet2photo', 'cezanne2photo', 'ukiyoe2photo',
                        'vangogh2photo', 'maps', 'cityscapes',
                        'facades', 'iphone2dslr_flower']

        assert self.dataset_name in dataset_list, "Dataset name not a valid member of " + ','.join(dataset_list)

        load_data = os.path.join('cycle_gan', self.dataset_name)

        dataset = tfds.load(load_data, as_supervised = True)
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
        image = tf.image.decode_jpeg(image)

        self.channels = image.shape[-1]

        image = self._random_jitter(image)
        image = self._normalize(image)

        return image


    def _load__test_image(self, filename):

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)

        image = self._normalize(image)

        return image

    def _load_custom_data(self):

        assert os.path.exists(os.path.join(self.datadir, '/trainA')), "trainA directory not found"
        train_data = tf.data.Dataset.list_files(os.path.join(self.datadir, '/trainA/*.jpg'))
        trainA = train_data.map(self._load__train_image)

        assert os.path.exists(os.path.join(self.datadir, '/trainB')), "trainB directory not found"
        train_data = tf.data.Dataset.list_files(os.path.join(self.datadir, '/trainB/*.jpg'))
        trainB = train_data.map(self._load__train_image)

        assert os.path.exists(os.path.join(self.datadir, '/testA')), "testA directory not found"
        test_data = tf.data.Dataset.list_files(os.path.join(self.datadir, '/testA/*.jpg'))
        testA = test_data.map(self._load__test_image)

        assert os.path.exists(os.path.join(self.datadir, '/testB')), "testB directory not found"
        test_data = tf.data.Dataset.list_files(os.path.join(self.datadir, '/testB/*.jpg'))
        testB = test_data.map(self._load__test_image)

        return trainA, trainB, testA, testB

    def load_dataset(self):

        assert self.dataset_name != None or self.datadir != None, "Enter directory to load custom data or choose from exisisting data to load from"

        if(self.dataset_name != None):
            trainA, trainB, testA, testB = self._load_cyclegan_data()

        else:
            trainA, trainB, testA, testB = self._load_custom_data()

        return trainA, trainB, testA, testB