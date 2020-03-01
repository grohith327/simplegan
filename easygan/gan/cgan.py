import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ..datasets.load_mnist import load_mnist_with_labels
from ..datasets.load_cifar10 import load_cifar10_with_labels
from ..datasets.load_custom_data import load_custom_data_with_labels
from ..losses.minmax_loss import gan_discriminator_loss, gan_generator_loss
import datetime

class CGAN:

    def __init__(self):

        self.image_size = None
        self.embed_dim = None
        self.n_classes = None

    def load_data(self, data_dir = None, use_mnist=False, use_cifar10=False,
        batch_size = 32, img_shape = (64, 64)):

        if(use_mnist):
            
            train_data, train_labels = load_mnist_with_labels()
            self.n_classes=10

        elif(use_cifar10):

            train_data, train_labels = load_cifar10_with_labels()
            self.n_classes=10

        else:

            train_data, train_labels = load_custom_data_with_labels(data_dir)
            self.n_classes = np.unique(train_labels).shape[0]

        self.image_size = train_data[0].shape

        train_data = (train_data - 127.5) / 127.5
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(10000).batch(batch_size)

        return train_ds

    def generator(self, params):

        noise_dim = params['noise_dim'] if 'noise_dim' in params else 100
        self.noise_dim = noise_dim
        gen_channels = params['gen_channels'] if 'gen_channels' in params else [128, 128]
        gen_layers = params['gen_layers'] if 'gen_layers' in params else 2
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (5, 5)
        self.embed_dim = params['embed_dim'] if 'embed_dim' in params else 100

        assert len(gen_channels) == gen_layers, "Dimension mismatch: length of generator channels should match number of generator layers"

        z = layers.Input(shape=self.noise_dim)
        label = layers.Input(shape=1)

        start_image_size = (self.image_size[0]//4, self.image_size[1]//4)

        embedded_label = layers.Embedding(input_dim=10, output_dim=self.embed_dim)(label)
        embedded_label = layers.Dense(units=start_image_size[0]*start_image_size[1],
                        activation=activation, kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer, input_dim=self.embed_dim)(embedded_label)
        embedded_label = layers.Reshape((start_image_size[0], start_image_size[1], 1))(embedded_label)

        input_img = layers.Dense(start_image_size[0]*start_image_size[1]*128)(z)
        input_img = layers.Reshape((start_image_size[0], start_image_size[1], 128))(input_img)

        x = layers.Concatenate()([input_img, embedded_label])

        # Upsampling
        for i in range(gen_layers):
            x = layers.Conv2DTranspose(filters=gen_channels[i], kernel_size=kernel_size, strides=(2, 2), padding="same", use_bias=False,
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        output = layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False, activation='tanh',
                                        kernel_initializer=kernel_initializer, 
                                        kernel_regularizer=kernel_regularizer)(x)
        model = tf.keras.Model([z, label], output)
        return model

    
    def discriminator(self, params):

        dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 0.4
        disc_channels = params['disc_channels'] if 'disc_channels' in params else [128,128]
        disc_layers = params['disc_layers'] if 'disc_layers' in params else 2
        activation = params['activation'] if 'activation' in params else 'relu'
        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'glorot_uniform'
        kernel_regularizer = params['kernel_regularizer'] if 'kernel_regularizer' in params else None
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (5, 5)

        assert len(disc_channels) == disc_layers, "Dimension mismatch: length of discriminator channels should match number of discriminator layers"

        input_image = layers.Input(shape=self.image_size)
        input_label = layers.Input(shape=1)
        embedded_label = layers.Embedding(input_dim=self.n_classes,output_dim=self.embed_dim)(input_label)
        embedded_label = layers.Dense(units=self.image_size[0]*self.image_size[1])(embedded_label)
        embedded_label = layers.Reshape((self.image_size[0],self.image_size[1],1))(embedded_label)
        
        x = layers.Concatenate()([input_image,embedded_label])

        for i in range(disc_layers):
            x = layers.Conv2D(filters=disc_channels[i],kernel_size=kernel_size,strides=(2, 2),padding='same',
                            kernel_initializer = kernel_initializer, 
                            kernel_regularizer = kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)
        fe = layers.Dropout(dropout_rate)(x)
        out_layer = layers.Dense(1, activation='sigmoid')(fe)

        model=tf.keras.Model(inputs=[input_image,input_label],outputs=out_layer)
        return model
  
    '''
    call build_model() to get the generator and discriminator objects
    '''
    def build_model(self, params = {'gen_layers':3, 'disc_layers':3, 'noise_dim':100, 'dropout_rate':0.4, 
        'activation':'relu', 'kernel_initializer': 'glorot_uniform', 'kernel_size':(5,5),
        'gen_channels': [64, 32, 16], 'disc_channels': [16, 32, 64], 'kernel_regularizer': None,
        'embed_dim':100}):
        
        self.gen_model, self.disc_model = self.generator(params), self.discriminator(params)

    def fit(self, train_ds = None, epochs = 100, gen_optimizer = 'Adam', disc_optimizer = 'Adam', 
        print_steps = 100, gen_learning_rate = 0.0001, disc_learning_rate = 0.0002, beta_1 = 0.5, 
        tensorboard = False, save_model = None):
        
        assert train_ds != None, 'Initialize training data through train_ds parameter'

        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        if(gen_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_learning_rate
        if(disc_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        disc_optimizer = getattr(tf.keras.optimizers, disc_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        for epoch in range(epochs):
            for data,labels in train_ds:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    noise = tf.random.normal([data.shape[0], self.noise_dim])
                    fake_imgs = self.gen_model([noise, labels])
                    sampled_labels = np.random.randint(0, 10, data.shape[0]).reshape(-1, 1)

                    real_output = self.disc_model([data,labels], training=True)
                    fake_output = self.disc_model([fake_imgs,sampled_labels], training=True)
              
                    G_loss = gan_generator_loss(fake_output)
                    D_loss = gan_discriminator_loss(real_output, fake_output)
            
                gradients_of_generator = gen_tape.gradient(G_loss, self.gen_model.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(D_loss, self.disc_model.trainable_variables)

                gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen_model.trainable_variables))
                disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc_model.trainable_variables))

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

    def generate_samples(self, n_samples = 1, labels_list = None, save_dir = None):

        assert labels_list is not None, "Enter list of labels to condition the generator"
        assert len(labels_list) == n_samples, "Number of samples does not match length of labels list"

        Z = np.random.uniform(-1, 1, (n_samples, self.noise_dim))
        labels_list = np.array(labels_list)
        generated_samples = self.gen_model([Z, labels_list])

        if(save_dir is None):
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            cv2.imwrite(os.path.join(save_dir, 'sample_' + str(i) + '.jpg'), sample)