simplegan.losses
================

.. currentmodule:: simplegan.losses

The ``losses`` sub-module provides users with useful loss functions which can be used to train their models.

.. contents::
    :local:

VanillaAutoencoder losses
-------------------------

.. autoclass:: mse_loss()
    :members:

GAN losses
----------

.. autoclass:: gan_discriminator_loss()
    :members:

.. autoclass:: gan_generator_loss()
    :members:

InfoGAN losses
--------------

.. autoclass:: auxillary_loss()
    :members:

Pix2Pix losses
---------------

.. autoclass:: pix2pix_discriminator_loss()
    :members:

.. autoclass:: pix2pix_generator_loss()
    :members:

CycleGAN losses
---------------

.. autoclass:: cycle_loss()
    :members:

.. autoclass:: identity_loss()
    :members:

Wasserstein losses
------------------

.. autoclass:: wgan_discriminator_loss()
    :members:

.. autoclass:: wgan_generator_loss()
    :members: