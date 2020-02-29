import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, BatchNormalization, LeakyReLU, Conv2DTranspose, ZeroPadding2D, Dense, Reshape, Flatten, ReLU, Input
from tensorflow.keras import Model
import numpy as np 
import datetime 
from ..datasets import pix2pix_dataloader
from ..losses import pix2pix_generator_loss, pix2pix_discriminator_loss
import cv2

'''
Paper: https://arxiv.org/abs/1611.07004 

The following code is inspired from: https://www.tensorflow.org/tutorials/generative/pix2pix#load_the_dataset

During trainig, samples will be saved at ./samples and saved rate at a rate given by save_img_per_epoch
'''

class Pix2Pix():

    def __init__(self):

        self.gen_model = None 
        self.disc_model = None
        self.channels = None
        self.LAMBDA = None
        self.img_size = None
        self.save_img_dir = None


    def load_data(self, data_dir = None, use_cityscapes = False, 
        use_edges2handbags = False, use_edges2shoes = False, 
        use_facades = False, use_maps = False, batch_size = 32):

        
        if(use_cityscapes):
            data_obj = pix2pix_dataloader(dataset_name = 'cityscapes')

        elif(use_edges2handbags):
            data_obj = pix2pix_dataloader(dataset_name = 'edges2handbags')

        elif(use_edges2shoes):
            data_obj = pix2pix_dataloader(dataset_name = 'edges2shoes')

        elif(use_facades):
            data_obj = pix2pix_dataloader(dataset_name = 'facades')

        elif(use_maps):
            data_obj = pix2pix_dataloader(dataset_name = 'maps')

        else:
            data_obj = pix2pix_dataloader(datadir = data_dir)

        train_ds, test_ds = data_obj.load_dataset()

        train_ds = train_ds.shuffle(100000).batch(batch_size)
        test_ds = test_ds.shuffle(100000).batch(batch_size)

        for data in train_ds:
            self.img_size = data[0].shape
            self.channels = data[0].shape[-1]
            break

        return train_ds, test_ds


    def _downsample(self, filters, kernel_size, batchnorm = True,
        kernel_initializer):

        model = tf.keras.Sequential()
        model.add(Conv2D(filters, kernel_size = kernel_size, strides = 2,
            kernel_initializer = kernel_initializer, padding = 'same', use_bias = False))

        if(batchnorm):
            model.add(BatchNormalization())

        model.add(LeakyReLU())

        return model

    def _upsample(self, filters, kernel_size, kernel_initializer,
        dropout = False, dropout_rate):

        model = tf.keras.Sequential()
        model.add(Conv2DTranspose(filters, kernel_size = kernel_size, 
            strides = 2, kernel_initializer = kernel_initializer, use_bias = False))
        model.add(BatchNormalization())

        if(dropout):
            model.add(Dropout(dropout_rate))

        model.add(ReLU())

        return model


    def generator(self, params):

        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'RandomNormal'
        dropout_rate = params['dropout_rate'] if 'dropout_rate' in params else 0.5
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (4, 4)
        gen_enc_layers = params['gen_enc_layers'] if 'gen_enc_layers' in params else 7
        gen_dec_layers = params['gen_dec_layers'] if 'gen_dec_layers' in params else 7
        gen_enc_channels = params['gen_enc_channels'] if 'gen_enc_channels' in params else [128, 256, 512, 512, 512, 512, 512]
        gen_dec_channels = params['gen_dec_channels'] if 'gen_dec_channels' in params else [512, 512, 512, 512, 256, 128, 64]

        assert len(gen_enc_channels) == gen_enc_layers, "Dimension mismatch: length of generator encoder channels should match number of generator encoder layers"
        assert len(gen_dec_channels) == gen_dec_layers, "Dimension mismatch: length of generator decoder channels should match number of generator decoder layers"

        inputs = Input(shape = self.img_size)

        down_stack = [self._downsample(64, 4, batchnorm = False, kernel_initializer)]

        for channel in gen_enc_channels:
            down_stack.append(self._downsample(channel, kernel_size, kernel_initializer = kernel_initializer))


        up_stack = []
        for i, channel in enumerate(gen_dec_channels):
            if(i < 2):
                up_stack.append(self._upsample(channel, kernel_size, kernel_initializer = kernel_initializer, 
                            dropout = True, dropout_rate = dropout_rate))
            else:
                up_stack.append(self._upsample(channel, kernel_size, kernel_initializer = kernel_initializer, 
                            dropout_rate = dropout_rate))

        last = Conv2DTranspose(self.channels, strides = 2, padding = 'same',
                                kernel_initializer = kernel_initializer, activation = 'tanh')

        x = inputs

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)

        model = Model(input = inputs, outputs = x)
        return model


    def discriminator(self, params):

        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'RandomNormal'
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (4, 4)
        disc_layers = params['disc_layers'] if 'disc_layers' in params else 4
        disc_channels = params['disc_channels'] if 'disc_channels' in params else [64, 128, 256, 512]

        assert len(disc_channels) == disc_layers, "Dimension mismatch: length of discriminator channels should match number of discriminator layers"

        inputs = Input(shape = self.img_size)
        target = Input(shape = self.img_size)

        x = Concatenate()[inputs, target]

        down_stack = []
        for i, channel in enumerate(disc_channels[:-1]):
            if(i == 0):
                down_stack.append(self._downsample(channel, kernel_size = kernel_size,
                                batchnorm = False, kernel_initializer = kernel_initializer))
            else:
                down_stack.append(self._downsample(channel, kernel_size = kernel_size,
                                kernel_initializer = kernel_initializer))


        down_stack.append(ZeroPadding2D())
        down_stack.append(Conv2D(disc_channels[-1], kernel_size = kernel_size,
                                strides = 1, kernel_initializer = kernel_initializer, 
                                use_bias = False))

        down_stack.append(BatchNormalization())
        down_stack.append(LeakyReLU())
        down_stack.append(ZeroPadding2D())

        last = Conv2D(1, kernel_size = kernel_size, strides = 1, 
                    kernel_initializer = kernel_initializer)

        for down in down_stack:
            x = down(x)

        out = last(x)
        model = Model(inputs = [inputs, target], outputs = out)
        return model


    def build_model(self, params = {'kernel_initializer': 'RandomNormal', 'dropout_rate':0.5,
        'gen_enc_layers': 7, 'kernel_size': (4, 4), 'gen_enc_channels': [128, 256, 512, 
        512, 512, 512, 512], 'gen_dec_layers': 7, 'gen_dec_channels': [512, 512, 512, 512,
        256, 128, 64], 'disc_layers': 4, 'disc_channels': [64, 128, 256, 512]}):

        '''
        Call build_model to initialize the generator and discriminator model

        generator -> U-Net
        discriminator -> PatchGAN 
        
        '''

        self.gen_model, self.disc_model = self.generator(params), self.discriminator(params)


    def _save_samples(self, model, ex_input, ex_target, count):

        assert os.path.exists(self.save_img_dir), "sample directory does not exist"

        prediction = model(ex_input, training = False)

        input_image = ex_input.numpy()
        target_image = ex_target.numpy()
        prediction = prediction.numpy()

        curr_dir = os.path.join(self.save_img_dir, count)
        os.mkdir(curr_dir)

        cv2.imwrite(os.path.join(curr_dir,'input_image.jpg'), input_image)
        cv2.imwrite(os.path.join(curr_dir,'target_image.jpg'), target_image)
        cv2.imwrite(os.path.join(curr_dir,'prediction.jpg'), prediction)


    def fit(self, train_ds = None, test_ds = None, epochs = 150, gen_optimizer = 'Adam', disc_optimizer = 'Adam', 
        print_steps = 100, gen_learning_rate = 2e-4, disc_learning_rate = 2e-4, beta_1 = 0.5, 
        tensorboard = False, save_model = None, LAMBDA = 100, save_img_per_epoch = 30):

        assert train_ds != None, 'Initialize training data through train_ds parameter'
        assert test_ds != None, 'Initialize testing data through test_ds parameter'
        self.LAMBDA = LAMBDA

        kwargs = {}
        kwargs['learning_rate'] = gen_learning_rate
        if(gen_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        gen_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_learning_rate
        if(gen_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        disc_optimizer = getattr(tf.keras.optimizers, gen_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        curr_dir = os.getcwd()
        os.mkdir(os.path.join(curr_dir, 'samples'))
        self.save_img_dir = os.path.join(curr_dir, 'samples')

        for epoch in range(epochs):

            for input_image, target_image in train_ds:

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    gen_output = self.gen_model(input_image, training = True)

                    disc_real_output = self.disc_model([input_image, target_image], training = True)
                    disc_gen_output = self.disc_model([input_image, gen_output], training = True)

                    gen_total_loss, gan_loss, l1_loss = pix2pix_generator_loss(disc_gen_output, gen_output, target_image, self.LAMBDA)
                    disc_loss = pix2pix_discriminator_loss(disc_real_output, disc_gen_output)

                gen_gradients = gen_tape.gradient(gen_total_loss, self.gen_model.trainable_variables)
                gen_optimizer.apply_gradients(zip(gen_gradients, self.gen_model.trainable_variables))

                disc_gradients = disc_tape.gradient(disc_loss, self.disc_model.trainable_variables)
                disc_optimizer.apply_gradients(zip(disc_gradients, self.disc_model.trainable_variables))


                if(steps % print_steps == 0):
                    print('Step:', steps+1, 'D_loss:', disc_loss.numpy(), 'G_loss', gen_total_loss.numpy())

                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('discr_loss', disc_loss.numpy(), step=steps)
                        tf.summary.scalar('total_gen_loss', gen_total_loss.numpy(), step=steps)
                        tf.summary.scalar('gan_loss', gan_loss.numpy(), step=steps)
                        tf.summary.scalar('l1_loss', l1_loss.numpy(), step=steps)


            if(epoch % save_img_per_epoch == 0):
                for input_image, target_image in test_ds.take(1):
                    self._save_samples(self.gen_model, input_image, target_image, str(epoch))


        if(save_model != None):

            assert type(save_model) == str, "Not a valid directory"
            if(save_model[-1] != '/'):
                self.gen_model.save_weights(save_model + '/generator_checkpoint')
                self.disc_model.save_weights(save_model + '/discriminator_checkpoint')
            else:
                self.gen_model.save_weights(save_model + 'generator_checkpoint')
                self.disc_model.save_weights(save_model + 'discriminator_checkpoint')