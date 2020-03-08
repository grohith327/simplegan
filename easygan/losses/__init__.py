from .minmax_loss import *
from .wasserstein_loss import *
from .mse_loss import *
from .vae_loss import *
from .pix2pix_loss import *
from .cyclegan_loss import *
from .infogan_loss import *
__all__ = ['gan_discriminator_loss',
           'gan_generator_loss',
           'mse_loss',
           'vae_loss',
           'wgan_discriminator_loss',
           'wgan_generator_loss',
           'pix2pix_generator_loss',
           'pix2pix_discriminator_loss',
           'cycle_loss',
           'identity_loss',
           'auxillary_loss']
