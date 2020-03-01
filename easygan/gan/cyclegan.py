import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, BatchNormalization, LeakyReLU, Conv2DTranspose, ZeroPadding2D, Dense, Reshape, Flatten, ReLU, Input
from tensorflow.keras import Model
import numpy as np 
import datetime 
import cv2 
from ..losses.minmax_loss import gan_generator_loss, gan_discriminator_loss
from ..losses.cyclegan_loss import cycle_loss, identity_loss
from ..datasets.load_cyclegan_datasets import cyclegan_dataloader
from .pix2pix import Pix2Pix


'''
Paper: https://arxiv.org/abs/1703.10593

Code inspired from: https://www.tensorflow.org/tutorials/generative/cyclegan#import_and_reuse_the_pix2pix_models
'''

class CycleGAN(Pix2Pix):

    def __init__(self):
        Pix2Pix.__init__(self)
        self.gen_model_g = None
        self.gen_model_f = None
        self.disc_model_x = None
        self.disc_model_y = None

    def load_data(self, data_dir = None, batch_size = 1, use_apple2orange = False,
                use_summer2winter_yosemite = False, use_horse2zebra = False,
                use_monet2photo = False, use_cezanne2photo = False, use_ukiyoe2photo = False, 
                use_vangogh2photo = False, use_maps = False, use_cityscapes = False,
                use_facades = False, use_iphone2dslr_flower = False):

        if(use_apple2orange):

            data_obj = cyclegan_dataloader(dataset_name = 'apple2orange')

        elif(use_summer2winter_yosemite):

            data_obj = cyclegan_dataloader(dataset_name = 'summer2winter_yosemite')

        elif(use_horse2zebra):

            data_obj = cyclegan_dataloader(dataset_name = 'horse2zebra')

        elif(use_monet2photo):

            data_obj = cyclegan_dataloader(dataset_name = 'monet2photo')

        elif(use_cezanne2photo):

            data_obj = cyclegan_dataloader(dataset_name = 'use_cezanne2photo')

        elif(use_ukiyoe2photo):

            data_obj = cyclegan_dataloader(dataset_name = 'ukiyoe2photo')

        elif(use_vangogh2photo):

            data_obj = cyclegan_dataloader(dataset_name = 'vangogh2photo')

        elif(use_maps):

            data_obj = cyclegan_dataloader(dataset_name = 'maps')

        elif(use_cityscapes):

            data_obj = cyclegan_dataloader(dataset_name = 'cityscapes')

        elif(use_facades):

            data_obj = cyclegan_dataloader(dataset_name = 'facades')

        elif(use_iphone2dslr_flower):

            data_obj = cyclegan_dataloader(dataset_name = 'iphone2dslr_flower')

        else:

            data_obj = cyclegan_dataloader(datadir = datadir)

        trainA, trainB, testA, testB = data_obj.load_dataset()

        trainA = trainA.shuffle(100000).batch(batch_size)
        trainB = trainB.shuffle(100000).batch(batch_size)

        testA = testA.shuffle(100000).batch(batch_size)
        testB = testB.shuffle(100000).batch(batch_size)

        for data in trainA.take(1):
            self.img_size = data[0].shape
            self.channels = data[0].shape[-1]

        return trainA, trainB, testA, testB


    def discriminator(self, params):

        kernel_initializer = params['kernel_initializer'] if 'kernel_initializer' in params else 'RandomNormal'
        kernel_size = params['kernel_size'] if 'kernel_size' in params else (4, 4)
        disc_layers = params['disc_layers'] if 'disc_layers' in params else 4
        disc_channels = params['disc_channels'] if 'disc_channels' in params else [64, 128, 256, 512]

        assert len(disc_channels) == disc_layers, "Dimension mismatch: length of discriminator channels should match number of discriminator layers"

        inputs = Input(shape = self.img_size)
        x = inputs

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
        model = Model(inputs = inputs, outputs = out)
        return model


    def build_model(self, params = {'kernel_initializer': 'RandomNormal', 'dropout_rate':0.5,
        'gen_enc_layers': 7, 'kernel_size': (4, 4), 'gen_enc_channels': [128, 256, 512, 
        512, 512, 512, 512], 'gen_dec_layers': 7, 'gen_dec_channels': [512, 512, 512, 512,
        256, 128, 64], 'disc_layers': 4, 'disc_channels': [64, 128, 256, 512]}):

        '''
        Call build model to initialize the two generators and discriminators

        Note: Forward and backward GANs have the same architecture
        '''

        self.gen_model_g, self.gen_model_f = self.generator(params), self.generator(params)
        self.disc_model_x, self.disc_model_y = self.discriminator(params), self.discriminator(params)


    def _save_samples(self, model, image, count):

        assert os.path.exists(self.save_img_dir), "sample directory does not exist"

        pred = model(image, training = False)
        pred = pred[0].numpy()
        image = image[0].numpy()

        curr_dir = os.path.join(self.save_img_dir, count)
        os.mkdir(curr_dir)

        cv2.imwrite(os.path.join(curr_dir,'input_image.jpg'), image)
        cv2.imwrite(os.path.join(curr_dir,'translated_image.jpg'), pred)


    def fit(self, trainA = None, trainB = None, testA = None, testB = None, epochs = 150, 
        gen_g_optimizer = 'Adam', gen_f_optimizer = 'Adam', disc_x_optimizer = 'Adam', disc_y_optimizer = 'Adam', 
        print_steps = 100, gen_g_learning_rate = 2e-4, gen_f_learning_rate = 2e-4,
        disc_x_learning_rate = 2e-4, disc_y_learning_rate = 2e-4, beta_1 = 0.5, tensorboard = False, 
        save_model = None, LAMBDA = 100, save_img_per_epoch = 30):


        assert trainA != None, 'Initialize training data A through trainA parameter'
        assert trainB != None, 'Initialize training data B through trainB parameter'
        assert testA != None, 'Initialize testing data A through testA parameter'
        assert testB != None, 'Initialize testing data B through testB parameter'
        self.LAMBDA = LAMBDA

        kwargs = {}
        kwargs['learning_rate'] = gen_g_learning_rate
        if(gen_g_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        gen_g_optimizer = getattr(tf.keras.optimizers, gen_g_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = gen_f_learning_rate
        if(gen_f_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        gen_f_optimizer = getattr(tf.keras.optimizers, gen_f_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_x_learning_rate
        if(disc_x_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        disc_x_optimizer = getattr(tf.keras.optimizers, disc_x_optimizer)(**kwargs)

        kwargs = {}
        kwargs['learning_rate'] = disc_y_learning_rate
        if(disc_y_optimizer == 'Adam'):
            kwargs['beta_1'] = beta_1
        disc_y_optimizer = getattr(tf.keras.optimizers, disc_y_optimizer)(**kwargs)

        if(tensorboard):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        steps = 0

        curr_dir = os.getcwd()
        os.mkdir(os.path.join(curr_dir, 'samples'))
        self.save_img_dir = os.path.join(curr_dir, 'samples')

        for epoch in range(epochs):

            for image_x, image_y in tf.data.Dataset.zip((trainA, trainB)):

                with tf.GradientTape(persistent = True) as tape:

                    fake_y = self.gen_model_g(image_x, training = True)
                    cycled_x = self.gen_model_f(fake_y, training = True)

                    fake_x = self.gen_model_f(image_y, training = True)
                    cycled_y = self.gen_model_g(fake_x, training = True)

                    same_x = self.gen_model_f(image_x, training = True)
                    same_y = self.gen_model_g(image_y, training = True)

                    disc_real_x = self.disc_model_x(image_x, training = True)
                    disc_real_y = self.disc_model_y(image_y, training = True)

                    disc_fake_x = self.disc_model_x(fake_x, training = True)
                    disc_fake_y = self.disc_model_y(fake_y, training = True)

                    gen_g_loss = gan_generator_loss(disc_fake_y)
                    gen_f_loss = gan_generator_loss(disc_fake_x)

                    total_cycle_loss = cycle_loss(image_x, cycled_x, self.LAMBDA) + cycle_loss(image_y, cycled_y, self.LAMBDA)

                    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(image_y, same_y, self.LAMBDA)
                    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(image_x, same_x, self.LAMBDA)

                    disc_x_loss = gan_discriminator_loss(disc_real_x, disc_fake_x)
                    disc_y_loss = gan_discriminator_loss(disc_real_y, disc_fake_y)

                generator_g_gradients = tape.gradient(total_gen_g_loss, self.gen_model_g.trainable_variables)
                generator_f_gradients = tape.gradient(total_gen_f_loss, self.gen_model_f.trainable_variables)
  
                discriminator_x_gradients = tape.gradient(disc_x_loss, self.disc_model_x.trainable_variables)
                discriminator_y_gradients = tape.gradient(disc_y_loss, self.disc_model_y.trainable_variables)

                gen_g_optimizer.apply_gradients(zip(generator_g_gradients, self.gen_model_g.trainable_variables))
                gen_f_optimizer.apply_gradients(zip(generator_f_gradients, self.gen_model_f.trainable_variables))
  
                disc_x_optimizer.apply_gradients(zip(discriminator_x_gradients, self.disc_model_x.trainable_variables))
                disc_y_optimizer.apply_gradients(zip(discriminator_y_gradients, self.disc_model_y.trainable_variables))

                if(steps % print_steps == 0):
                    print('Step:', steps+1, 'Generator_G_loss:', total_gen_g_loss.numpy(),
                        'Generator_F_loss:', total_gen_f_loss.numpy(),
                        'Discriminator_X_loss:', disc_x_loss.numpy(), 
                        'Discriminator_Y_loss:', disc_y_loss.numpy())


                steps += 1

                if(tensorboard):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Generator_G_loss', total_gen_g_loss.numpy(), step=steps)
                        tf.summary.scalar('Generator_F_loss', total_gen_f_loss.numpy(), step=steps)
                        tf.summary.scalar('Discriminator_X_loss', disc_x_loss.numpy(), step=steps)
                        tf.summary.scalar('Discriminator_Y_loss', disc_y_loss.numpy(), step=steps)


            if(epoch % save_img_per_epoch == 0):
                for image in testA.take(1):
                    self._save_samples(self.gen_model_g, image)

        if(save_model != None):

            assert type(save_model) == str, "Not a valid directory"
            if(save_model[-1] != '/'):
                self.gen_model_g.save_weights(save_model + '/generator_g_checkpoint')
            else:
                self.gen_model_g.save_weights(save_model + 'generator_g_checkpoint')


    def generate_samples(self, test_ds = None, save_dir = None):

        assert test_ds is not None, "Enter input test dataset"

        generated_samples = []
        for image in test_ds:
            gen_image = self.gen_model_g(image, training = False).numpy()
            generated_samples.append(gen_image)

        if(save_dir is None):
            return generated_samples

        assert os.path.exists(save_dir), "Directory does not exist"
        for i, sample in enumerate(generated_samples):
            cv2.imwrite(os.path.join(save_dir, 'sample_' + str(i) + '.jpg'), sample)