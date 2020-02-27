import tensorflow as tf 
from tensorflow.keras.layers import Dropout, BatchNormalization, Lambda, Dense, Reshape, Input, ReLU, Conv2D, Conv2DTranspose, Embedding
from tensorflow.keras import Model
import numpy as np
from ..datasets import load_cifar10
import datetime
from ..losses import mse_loss

'''
vector quantized vae

Reference: https://arxiv.org/abs/1711.00937


The code is inspired by the following sources:
-> https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
-> https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
-> https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
'''

class VectorQuantizer(Model):

    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        intializer = tf.keras.initializers.VarianceScaling(distribution = 'uniform')
        self.embedding = Embedding(self.num_embeddings, self.embedding_dim,
            embeddings_initializer = initializer)

    def call(self, x):

        x = tf.tranpose(x, perm = [0, 2, 3, 1])
        flat_x = tf.reshape(x, [-1, self.embedding_dim])

        distances = (tf.math.reduce_sum(flat_x**2, axis=1, keepdims=True)  - 
            2*tf.linalg.matmul(flat_x, self.embedding.get_weights()[0]) +
            tf.math.reduce_sum(self.embedding.get_weights()[0]**2, axis=0, keepdims=True))

        encoding_indices = tf.math.argmax(-distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = tf.linalg.matmul(encodings, self.embedding.get_weights()[0])
        quantized = tf.reshape(quantized, x.shape)

        e_latent_loss = tf.math.reduce_mean((tf.stop_gradient(quantized) - x)**2)
        q_latent_loss = tf.math.reduce_mean((quantized - tf.stop_gradient(x))**2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + tf.stop_gradient(quantized - x)
        avg_probs = tf.math.reduce_mean(encodings, axis = 0)
        perplexity = tf.math.exp( - tf.math.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return loss, tf.tranpose(quantized, perm = [0, 3, 1, 2]), perplexity, encodings


class residual(Model):

    def __init__(self, num_hiddens, num_residual_layers, 
        num_residual_hiddens):
        super(residual, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self.relu = ReLU()
        self.conv1 = Conv2D(self.num_residual_hiddens, activation='relu',kernel_size = (3,3),
            strides = (1,1))

        self.conv2 = Conv2D(self.num_hiddens, kernel_size = (1,1), strides = (1,1))


    def call(self,x):

        for _ in range(self.num_residual_layers):

            output = self.relu(x)
            output = self.conv1(output)
            output = self.conv2(output)

            x += output

        x = self.relu(x)
        return x


class encoder(Model):

    def __init__(self, params):
        super(encoder, self).__init__()

        self.num_hiddens = params['num_hiddens'] 
        self.num_residual_hiddens = params['num_residual_hiddens']
        self.num_residual_layers = params['num_residual_layers']

        self.conv1 = Conv2D(self.num_hiddens // 2, kernel_size = (4,4), strides =(2,2), 
            activation = 'relu')

        self.conv2 = Conv2D(self.num_hiddens, kernel_size = (4,4), strides = (2,2), 
            activation='relu')

        self.conv3 = Conv2D(self.num_hiddens, kernel_size = (3,3), strides = (1,1))

        self.residual_stack = residual(self.num_hiddens, self.num_residual_layers, 
            self.num_residual_hiddens)

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_stack(x)

        return x


class decoder(Model):

    def __init__(self, params):
        super(decoder, self).__init__()

        self.num_hiddens = params['num_hiddens'] 
        self.num_residual_hiddens = params['num_residual_hiddens'] 
        self.num_residual_layers = params['num_residual_layers'] 


        self.conv1 = Conv2D(self.num_hiddens, kernel_size = (3,3), strides = (1,1))

        self.residual_stack = residual(self.num_hiddens, self.num_residual_layers, 
            self.num_residual_hiddens)

        self.upconv1 = Conv2DTranspose(self.num_hiddens //2, kernel_size = (4, 4), 
            strides = (2,2), activation = 'relu')

        self.upconv2 = Conv2DTranspose(self.image_size[-1], kernel_size = (4, 4), strides = (2, 2))


    def call(self, x):

        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.upconv1(x)
        x = self.upconv2(x)

        return x


class nn_model(Model):

    def __init__(self, params):
        super(nn_model, self).__init__()

        embedding_dim = params['embedding_dim'] if 'embedding_dim' in params else 64
        commiment_cost = params['commiment_cost'] if 'commiment_cost' in params else 0.25

        self.encoder = encoder(params)
        self.pre_vq_conv = Conv2D(embedding_dim, kernel_size = (1,1), strides = (1,1))
        self.decoder = decoder(params)
        self.vq_vae = VectorQuantizer(params['num_embeddings'], params['embedding_dim'], 
            params['commiment_cost'])


    def call(self, x):

        output = self.encoder(x)
        output = self.pre_vq_conv(output)
        loss, quantized, perplexity = self.vq_vae(output)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity


class VQ_VAE():

    def __init__(self):

        self.image_size = None 
        self.model = None
        self.data_var = None

    def load_data(self, data_dir = None, use_mnist = False, 
        use_cifar10 = False, batch_size = 32, img_shape = (64, 64)):


        if(use_mnist):

            train_data = load_mnist()
        
        elif(use_cifar10):

            train_data = load_cifar10()

        else:

            train_data = load_custom_data(data_dir, img_shape)

        self.image_size = train_data.shape[1:]
        self.data_var = np.var(train_data / 255)

        train_data = (train_data / 255.0) - 0.5
        train_ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)

        return train_ds


    def build_model(self, params = {'num_hiddens':128, 'num_residual_hiddens': 32, 'num_residual_layers':2, 
        'embedding_dim':64, 'commiment_cost': 0.25}):


        assert 'num_hiddens' in params, "Enter num_hiddens parameter"
        assert 'num_residual_hiddens' in params, "Enter num_hiddens parameter"
        assert 'num_residual_layers' in params, "Enter num_hiddens parameter"

        self.model = nn_model(params)

    
    def fit(self, train_ds = None, epochs = 100, optimizer = 'Adam', print_steps = 100, 
        learning_rate = 3e-4, tensorboard = False, save_model = None):

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
                    vq_loss, data_recon, perplexity = self.model(data)
                    recon_err = mse_loss(data_recon, data) / self.data_var
                    loss = vq_loss + recon_err

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if(steps % print_steps == 0):
                    print('Step:', steps +1, 'total_loss:', loss.numpy(), 
                        'vq_loss:', vq_loss.numpy(), 'reconstruction loss:', recon_err.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('vq_loss', vq_loss.numpy(), step=steps)
                        tf.summary.scalar('reconstruction_loss', recon_err.numpy(), step=steps)



        if(save_model != None):

            assert type(save_model) == str, "Not a valid directory"
            if(save_model[-1] != '/'):
                self.model.save_weights(save_model + '/vq_vae_checkpoint')
            else:
                self.model.save_weights(save_model + 'vq_vae_checkpoint')