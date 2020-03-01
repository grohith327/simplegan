import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose, Dense, Reshape, Flatten, MaxPool2D
from tensorflow.keras import Model
import numpy as np
from ..datasets.load_cifar10 import load_cifar10
from ..datasets.load_mnist import load_mnist
from ..datasets.load_custom_data import load_custom_data
import datetime 
from ..losses.mse_loss import mse_loss

'''
vanilla_autoencoder imports from tensorflow Model class

Create an instance of the class and compile it by using the loss from ../losses/mse_loss and use an optimizer and metric of your choice

use the fit function to train the model. 
'''

class ConvolutionalAutoencoder():

    def __init__(self):
        '''
        initialize the number of encoder and layers
        '''
        super(ConvolutionalAutoencoder, self).__init__()
        self.model = tf.keras.Sequential()
        self.image_size = None


    def load_data(self, data_dir = None, use_mnist = False, 
        use_cifar10 = False, batch_size = 32, img_shape = (64,64)):

        '''
        choose the dataset, if None is provided returns an assertion error -> ../datasets/load_custom_data
        returns a tensorflow dataset loader
        '''
        if(use_mnist):

            train_data = load_mnist()

        elif(use_cifar10):

            train_data = load_cifar10()

        else:

            train_data = load_custom_data(data_dir)

        self.image_size = train_data.shape[1:]

        train_data = train_data / 255
        train_ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)

        return train_ds

    '''
    encoder and decoder layers for custom dataset can be reimplemented by inherting this class(vanilla_autoencoder)
    '''
    def encoder(self, params):

        enc_channels = params['enc_channels'] if 'enc_channels' in params else [32, 64]
        encoder_layers = params['encoder_layers'] if 'encoder_layers' in params else 2
        interm_dim = params['interm_dim'] if 'interm_dim' in params else 128
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (5, 5)

        assert len(enc_channels) == encoder_layers, "Dimension mismatch: length of enocoder encoder channels should match number of encoder layers"

        model = tf.keras.Sequential()

        model.add(Conv2D(enc_channels[0] // 2, kernel_size = kernel_size, padding='same', activation=activation, 
            kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, 
            input_shape = self.image_size))
        model.add(MaxPool2D())

        for i in range(encoder_layers):
            model.add(Conv2D(enc_channels[i], kernel_size = kernel_size, padding='same', activation=activation, 
            kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
            model.add(MaxPool2D())

        model.add(Flatten())
        model.add(Dense(interm_dim, activation='sigmoid'))

        return model


    def decoder(self, params):

        dec_channels = params['dec_channels'] if 'dec_channels' in params else [64, 32]
        decoder_layers = params['decoder_layers'] if 'decoder_layers' in params else 2
        interm_dim = params['interm_dim'] if 'interm_dim' in params else 128
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (5, 5)

        assert len(dec_channels) == decoder_layers, "Dimension mismatch: length of decoder channels should match number of decoder layers"

        model = tf.keras.Sequential()

        model.add(Dense((self.image_size[0] // 4)*(self.image_size[1] // 4)*(dec_channels[0]*2), 
            activation = activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer, input_dim = interm_dim))
        
        model.add(Reshape(((self.image_size[0] // 4), (self.image_size[1] // 4), (dec_channels[0]*2))))

        k = 0
        for _ in range(decoder_layers//2):
            model.add(Conv2DTranspose(dec_channels[k], kernel_size = kernel_size, strides=(1, 1), padding='same', 
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
            k += 1


        model.add(Conv2DTranspose(dec_channels[k], kernel_size = kernel_size, strides=(2, 2), padding='same', 
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
        

        for _ in range(decoder_layers//2):
            model.add(Conv2DTranspose(dec_channels[k], kernel_size = kernel_size, strides=(1, 1), padding='same', 
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
            k += 1


        model.add(Conv2DTranspose(self.image_size[2], kernel_size = kernel_size, strides = (2,2), padding='same', activation='tanh'))

        return model

    '''
    call build_model to intialize the layers before you train the model
    '''
    def build_model(self, params = {'encoder_layers':2, 'decoder_layers':2, 'interm_dim': 128, 
        'enc_channels': [32, 64], 'dec_channels':[64, 32], 'kernel_size':(5,5), 'activation':'relu', 
        'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None}):

        self.model.add(self.encoder())
        self.model.add(self.decoder())

    def fit(self, train_ds = None, epochs = 100, optimizer = 'Adam', print_steps = 100, 
        learning_rate = 0.001, tensorboard = False, save_model = None):

        assert train_ds != None, 'Initialize training data through train_ds parameter'

        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        for epoch in range(epochs):

            for data in train_ds:

                with tf.GradientTape() as tape:
                    recon_data = self.model(data)
                    loss = mse_loss(data, recon_data)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if(steps % print_steps == 0):
                    print("Step:", steps+1, 'reconstruction loss', loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss.numpy(), step=steps)


        if(save_model != None):

            assert type(save_model) == str, "Not a valid directory"
            if(save_model[-1] != '/'):
                self.model.save_weights(save_model + '/convolutional_autoencoder_autoencoder_checkpoint')
            else:
                self.model.save_weights(save_model + 'convolutional_autoencoder_autoencoder_checkpoint')


    def generate_samples(self, test_ds = None, save_dir = None):

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = []
        for data in test_ds:
            gen_sample = self.model(data, training = False)
            generated_samples.append(gen_sample)

        if(save_dir is None):
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            cv2.imwrite(os.path.join(save_dir, 'sample_' + str(i) + '.jpg'), sample)