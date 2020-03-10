Examples
========

``simplegan`` allows users to not only train generative models with few lines of code but also provdies some customizability options that let them train models on their own data or tweak the parameters of the default model. Have a look at the examples below that provides an overview of the functionality.

Vanilla Autoencoder
-------------------

In this example, we will use the mnist dataset to train a vanilla autoencoder and save the model.

.. code:: python

	from simplegan.autoencoder import VanillaAutoencoder
	autoenc = VanillaAutoencoder()
	train_ds, test_ds = autoenc.load_data(use_mnist = True)
	autoenc.fit(train_ds, test_ds, epochs = 10, save_model = './')

Using just three lines of code, we can train an Autoencoder model.

DCGAN
-----

In this example, let us build a DCGAN model with modified generator and discriminator architecture and train it on a custom local data directory.

.. code:: python

	from simplegan.gan import DCGAN
	gan = DCGAN(dropout_rate = 0.5, kernel_size = (4,4), gen_channels = [128, 64, 32])
	data = gan.load_data(data_dir = './', batch_size = 64, img_shape = (200, 200))
	gan.fit(data, epochs = 50, gen_optimizer = 'RMSprop', disc_learning_rate = 2e-3)
	generated_samples = gan.generate_samples(n_samples = 5)

Pix2Pix
-------

In this example, let us build a Pix2Pix model which a U-Net generator and a patchGAN discriminator. We train the pix2pix network on facades dataset.

.. code:: python

	from simplegan.gan import Pix2Pix
	gan = Pix2Pix()
	train_ds, test_ds = gan.load_data(use_facades = True, batch_size = 32)
	gan.fit(train_ds, test_ds, epochs = 200)

Have a look at the `examples <https://github.com/grohith327/simplegan/tree/master/examples>`_ directory which has notebooks to help you better understand on how to get started.