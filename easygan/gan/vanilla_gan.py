import tensorflow as tf 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from ..datasets import load_cifar10, load_mnist, load_custom_data
from ..losses import gan_discriminator_loss, gan_generator_loss
import numpy as np
import datetime

'''
vanilla gan imports from tensorflow Model class

Original GAN paper: https://arxiv.org/abs/1406.2661
'''
class VanillaGAN():

    def __init__(self):

        self.image_size = None
        self.noise_dim = None
        self.gen_model = None
        self.disc_model = None

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

        train_data = train_data.reshape((-1, self.image_size[0]*self.image_size[1]*self.image_size[2])) / 255
        train_ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(batch_size)

        return train_ds

    '''
    Create a child class to modify generator and discriminator architecture for 
    custom dataset
    '''

    def generator(self, params):

        noise_dim = params['noise_dim'] if 'noise_dim' in params else 64
        self.noise_dim = noise_dim
        dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 64
        gen_units = params['gen_units'] if 'gen_units' in params else [128, 256, 512]
        gen_layers = params['gen_layers'] if 'gen_layers' in params else 3
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None

        assert len(gen_units) == gen_layers, "Dimension mismatch: length of generator units should match number of generator layers"

        model = tf.keras.Sequential()

        model.add(Dense(gen_units[0] // 2, activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer, input_dim = noise_dim))
        model.add(Dropout(dropout_rate))

        for i in range(gen_layers):
            model.add(Dense(gen_units[i], activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout_rate))

        model.add(Dense(self.image_size[0]*self.image_size[1]*self.image_size[2], activation='sigmoid'))
        return model

    def discriminator(self, params):

        dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 0.4
        disc_units = params['disc_units'] if 'disc_units' in params else [512, 256, 128]
        disc_layers = params['disc_layers'] if 'disc_layers' in params else 3
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None

        assert len(disc_units) == disc_layers, "Dimension mismatch: length of generator units should match number of generator layers"

        model = tf.keras.Sequential()

        model.add(Dense(disc_units[0]*2, activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer, input_dim = self.image_size[0]*self.image_size[1]*self.image_size[2]))
        model.add(Dropout(dropout_rate))

        for i in range(disc_layers):
            model.add(Dense(disc_units[i], activation= activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))
        return model

    '''
    call build_model() to get the generator and discriminator objects
    '''
    def build_model(self, params = {'gen_layers':3, 'disc_layers':3, 'noise_dim':64, 'dropout_rate':0.4, 
        'activation':'relu', 'kernel_initializer': 'glorot_uniform', 
        'gen_units': [128, 256, 512], 'disc_units': [512, 256, 128], 'kernel_regularizer': None}):

        self.gen_model, self.disc_model =  self.generator(params), self.discriminator(params)

    def fit(self, train_ds = None, epochs = 100, gen_optimizer = 'Adam', disc_optimizer = 'Adam', 
        print_steps = 100, gen_learning_rate = 0.0001, disc_learning_rate = 0.0001, 
        tensorboard = False, save_model = None):

        assert train_ds != None, 'Initialize training data through train_ds parameter'

        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_learning_rate
        disc_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        for epoch in range(epochs):
            for data in train_ds:

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    real_logits = self.disc_model(real)
                    D_loss = gan_discriminator_loss(real_logits, fake_logits)

                gradients = tape.gradient(D_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(zip(gradients,self.disc_model.trainable_variables))

                with tf.GradientTape() as tape:

                    Z = np.random.uniform(-1, 1, (data.shape[0], self.noise_dim))
                    fake = self.gen_model(Z)
                    fake_logits = self.disc_model(fake)
                    G_loss = gan_generator_loss(fake_logits)

                gradients = tape.gradient(G_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(zip(gradients, self.gen_model.trainable_variables))


                if(steps % print_steps == 0):
                    print('Step:', steps+1, 'D_loss:', D_loss.numpy(), 'G_loss', G_loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('discr_loss', D_loss.numpy(), step=steps)
                        tf.summary.scalar('genr_loss', G_loss.numpy(), step=steps)


        if(save_model != None):

            assert type(save_model) == str, "Not a valid directory"
            if(save_model[-1] != '/'):
                self.gen_model.save_weights(save_model + '/generator_checkpoint')
                self.disc_model.save_weights(save_model + '/discriminator_checkpoint')
            else:
                self.gen_model.save_weights(save_model + 'generator_checkpoint')
                self.disc_model.save_weights(save_model + 'discriminator_checkpoint')