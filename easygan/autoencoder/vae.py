import tensorflow as tf 
from tensorflow.keras.layers import Dropout, BatchNormalization, Lambda, Dense, Reshape, Input
from tensorflow.keras import Model
import numpy as np
from ..datasets import load_cifar10, load_mnist, load_custom_data

'''
vae imports from tensorflow Model class

source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Reference: https://arxiv.org/abs/1312.6114

Create an instance of the class and compile it by using the loss from ../losses/vae_loss and use an optimizer and metric of your choice

use the fit function to train the model. 
'''

class VAE(Model):

	def __init__(self):

		super(VAE, self).__init__()
		self.enc = None
		self.dec = None
		self.image_size = None


	def load_data(self, data_dir = None, use_mnist = False, 
        use_cifar10 = False, batch_size = 32, img_shape = (64, 64)):

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

        train_data = train_data.reshape((-1, self.image_size[0]*self.image_size[1]*self.image_size[2])) / 255
        train_ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)

		return train_ds


	def sampling(self, [z_mean, z_var]):

		batch = tf.keras.backend.shape(z_mean)[0]
		dim = tf.keras.backend.int_shape(z_mean)[1]

		epsilon = tf.keras.backend.random_normal((batch, dim))

		return z_mean + tf.keras.backend.exp(0.5 * z_var) * epsilon

    '''
    encoder and decoder layers for custom dataset can be reimplemented by inherting this class(vae)
    '''

    def encoder(self, params):

		enc_units = params['enc_units'] if 'enc_units' in params else [256, 128]
        encoder_layers = params['encoder_layers'] if 'encoder_layers' in params else 2
        interm_dim = params['interm_dim'] if 'interm_dim' in params else 64
        latent_dim = params['latent_dim'] if 'latent_dim' in params else 32
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None

        assert len(enc_units) == encoder_layers, "Dimension mismatch: length of enocoder units should match number of encoder layers"

		inputs = Input(shape = self.image_size[0]*self.image_size[1]*self.image_size[2])
		outputs = Dense(enc_units[0] * 2, activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(inputs)

        for i in range(encoder_layers):
            outputs = Dense(enc_units[i], activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(outputs)
            
        x = Dense(interm_dim,activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(outputs)

        z_mean = Dense(latent_dim)(x)
        z_var = Dense(latent_dim)(x)

        ## Sampling from intermediate dimensiont to get a probability density 
        z = Lambda(self.sampling, output_shape = (latent_dim, ))([z_mean, z_var])

        model = Model(inputs, [z_mean, z_var])
        return model


    def decoder(self, params):

    	dec_units = params['dec_units'] if 'dec_units' in params else [128, 256]
        decoder_layers = params['decoder_layers'] if 'decoder_layers' in params else 2
        interm_dim = params['interm_dim'] if 'interm_dim' in params else 64
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        latent_dim = params['latent_dim'] if 'latent_dim' in params else 32

        assert len(dec_units) == decoder_layers, "Dimension mismatch: length of decoder units should match number of decoder layers"

    	inputs = Input(shape = latent_dim)
    	outputs = Dense(dec_units[0]  // 2, activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(inputs)

    	for i in range(decoder_layers):
    		outputs = Dense(dec_units[i], activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)(outputs)

    	outputs = Dense(self.image_size[0]*self.image_size[1]*self.image_size[2], activation = 'sigmoid')(outputs)

    	model = Model(inputs, outputs)
    	return model


    '''
    call build_model to intialize the layers before you train the model
    '''
	def build_model(self, params = {'encoder_layers':2, 'decoder_layers':2, 
        'enc_units': [256, 128], 'dec_units':[128, 256], 'interm_dim':256, 'latent_dim':32, 
        'activation':'relu', 'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None}):

        self.enc = self.encoder(params)
        self.dec = self.decoder(params)


	def call(self, x):

		x = self.enc(x)
		x = self.dec(x)
		return x












