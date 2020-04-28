# SimpleGAN

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE) [![Documentation Status](https://readthedocs.org/projects/simplegan/badge/?version=latest)](https://simplegan.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/simplegan)](https://pepy.tech/project/simplegan) [![Downloads](https://pepy.tech/badge/simplegan/month)](https://pepy.tech/project/simplegan/month) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Framework to ease training of generative models**

SimpleGAN is a framework based on [TensorFlow](https://www.tensorflow.org/) to make training of generative models easier. SimpleGAN provides high level APIs with customizability options to user which allows them to train a generative models with few lines of code or the user can reuse modules from the exisiting architectures to run custom training loops and experiments.
### Requirements
Make sure you have the following packages installed
* [tensorflow](https://www.tensorflow.org/install)
* [tqdm](https://github.com/tqdm/tqdm#latest-pypi-stable-release)
* [imagio](https://pypi.org/project/imageio/)
* [opencv](https://pypi.org/project/opencv-python/)
* [tensorflow-datasets](https://www.tensorflow.org/datasets/overview#installation)
### Installation
Latest stable release:
```bash
  $ pip install simplegan
```
Latest Development release:
```bash
  $ pip install git+https://github.com/grohith327/simplegan.git
```
### Getting Started
##### DCGAN
```python
from simplegan.gan import DCGAN

## initialize model
gan = DCGAN() 

## load train data
train_ds = gan.load_data(use_mnist = True)

## get samples from the data object
samples = gan.get_sample(train_ds, n_samples = 5)

## train the model
gan.fit(train_ds = train_ds)

## get generated samples from model
generated_samples = gan.generate_samples(n_samples = 5)
```
##### Custom training loops for GANs
```python
from simplegan.gan import Pix2Pix

## initialize model
gan = Pix2Pix()

## get generator module of Pix2Pix
generator = gan.generator() ## A tf.keras model

## get discriminator module of Pix2Pix
discriminator = gan.discriminator() ## A tf.keras model

## training loop
with tf.GradientTape() as tape:
""" Custom training loops """
```
##### Convolutional Autoencoder
```python
from simplegan.autoencoder import ConvolutionalAutoencoder

## initialize autoencoder
autoenc = ConvolutionalAutoencoder()

## load train and test data
train_ds, test_ds = autoenc.load_data(use_cifar10 = True)

## get sample from data object
train_sample = autoenc.get_sample(data = train_ds, n_samples = 5)
test_sample = autoenc.get_sample(data = test_ds, n_samples = 1)

## train the autoencoder
autoenc.fit(train_ds = train_ds, epochs = 5, optimizer = 'RMSprop', learning_rate = 0.002)

## get generated test samples from model
generated_samples = autoenc.generate_samples(test_ds = test_ds.take(1))
```
To have a look at more examples in detail, check [here](examples)
### Documentation
Check out the [docs page](https://simplegan.readthedocs.io/en/latest/)
### Provided models
| Model | Generated Images |
|:---------:|:--------------:|
| Vanilla Autoencoder | None |
| Convolutional Autoencoder | ![](https://github.com/grohith327/simplegan/blob/master/assets/mnist_conv_ae.png) |
| Variational Autoencoder [[Paper](https://arxiv.org/abs/1312.6114)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/vae.jpeg) |
| Vector Quantized - Variational Autoencoder [[Paper](https://arxiv.org/abs/1711.00937)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/vq_vae.png) |
| Vanilla GAN [[Paper](https://arxiv.org/abs/1406.2661)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/GAN.png) |
| DCGAN [[Paper](https://arxiv.org/abs/1511.06434)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/DCGAN.png) |
| WGAN [[Paper](https://arxiv.org/abs/1701.07875)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/WGAN.png) |
| CGAN [[Paper](https://arxiv.org/abs/1411.1784)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/CGAN.png) |
| InfoGAN [[Paper](https://arxiv.org/abs/1606.03657)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/InfoGAN.png) |
| Pix2Pix [[Paper](https://arxiv.org/abs/1611.07004)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/Pix2Pix.png) |
| CycleGAN [[Paper](https://arxiv.org/abs/1703.10593)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/CycleGAN.png) |
| 3DGAN(VoxelGAN) [[Paper](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/3DGAN.png) |
| Self-Attention GAN(SAGAN) [[Paper](https://arxiv.org/pdf/1805.08318.pdf)] | ![](https://github.com/grohith327/simplegan/blob/master/assets/SAGAN.png) |


### Contributing
We appreciate all contributions. If you are planning to perform bug-fixes, add new features or models, please file an issue and discuss before making a pull request.
### Citation
```
@software{simplegan,
    author = {{Rohith Gandhi et al.}},
    title = {simplegan},
    url = {https://simplegan.readthedocs.io},
    version = {0.2.8},
}
```
### Contributors 
* [Rohith Gandhi](https://github.com/grohith327)
* [Prem Kumar](https://github.com/Prem-kumar27)
