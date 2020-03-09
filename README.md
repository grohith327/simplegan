# SimpleGAN
**Framework to ease training of generative models**
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)

SimpleGAN is a framework based on [TensorFlow](https://www.tensorflow.org/) to make training of generative models easier. SimpleGAN provides high level APIs with customizability options to user which allows them to train a generative model with few lines of code.

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
  $ pip install git+https://github.com/grohith327/EasyGAN.git
```
### Getting Started
```python
>>> from simplegan.gan import DCGAN
>>> gan = DCGAN()
>>> train_ds = gan.load_data(use_mnist = True)
>>> samples = gan.get_sample(train_ds, n_samples = 5)
>>> gan.fit(train_ds = train_ds)
>>> generated_samples = gan.generate_samples(n_samples = 5)
```
To have a look at more examples, check [here](examples)
### Provided models
* Autoencoders
    * Vanilla Autoencoder
    * Convolutional Autoencoder
    * Variational Autoencoder [[Paper](https://arxiv.org/abs/1312.6114)]
    * Vector Quantized - Variational Autoencoder [[Paper](https://arxiv.org/abs/1711.00937)]
* Generative Adversarial Networks(GANs)
    * Vanilla GAN [[Paper](https://arxiv.org/abs/1406.2661)]
    * DCGAN [[Paper](https://arxiv.org/abs/1511.06434)]
    * WGAN [[Paper](https://arxiv.org/abs/1701.07875)]
    * CGAN [[Paper](https://arxiv.org/abs/1411.1784)]
    * InfoGAN [[Paper](https://arxiv.org/abs/1606.03657)]
    * Pix2Pix [[Paper](https://arxiv.org/abs/1611.07004)]
    * CycleGAN [[Paper](https://arxiv.org/abs/1703.10593)]

### Contributing
We appreciate all contributions. If you are planning to perform bug-fixes, add new features or models, please file an issue and discuss before making a pull request.
### Contributors 
* [Rohith Gandhi](https://github.com/grohith327)
* [Prem Kumar](https://github.com/Prem-kumar27)
