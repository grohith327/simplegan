import tensorflow as tf
from ..layers.spectralnorm import SpectralNormalization
from ..layers.conditionalbatchnorm import ConditionalBatchNorm

__all__ = ["GenResBlock", "DiscResBlock", "DiscOptResBlock"]


class GenResBlock(tf.keras.Model):
    def __init__(self, filters, n_classes, kernel_size=3, pad="same", spectral_norm=False):
        super(GenResBlock, self).__init__()

        self.cbn1 = ConditionalBatchNorm(n_classes)
        self.cbn2 = ConditionalBatchNorm(n_classes)

        if spectral_norm:
            self.deconv2a = SpectralNormalization(
                tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=pad)
            )

            self.deconv2b = SpectralNormalization(
                tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=pad)
            )
            self.shortcut_conv = SpectralNormalization(
                tf.keras.layers.Conv2DTranspose(filters, kernel_size=1, padding=pad)
            )
        else:
            self.deconv2a = tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=pad)

            self.deconv2b = tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=pad)
            self.shortcut_conv = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size=1, padding=pad
            )

        self.up_sample = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, inp, labels=None, training=False):

        x = self.cbn1(inp, labels)
        x = tf.nn.relu(x)
        x = self.up_sample(x)
        x = self.deconv2a(x)

        x = self.cbn2(x, labels)

        x = tf.nn.relu(x)
        x = self.deconv2b(x)

        sc_x = self.shortcut_conv(inp)
        sc_x = self.up_sample(sc_x)
        return x + sc_x


class DiscResBlock(tf.keras.Model):
    def __init__(
        self, filters, kernel_size=3, downsample=False, pad="same", spectral_norm=False
    ):
        super(DiscResBlock, self).__init__()

        if spectral_norm:
            self.conv1 = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
                )
            )
            self.conv2 = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
                )
            )
            self.shortcut_conv = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    kernel_initializer="glorot_uniform",
                    padding=pad,
                )
            )
        else:
            self.conv1 = tf.keras.layers.Conv2D(
                filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
            )
            self.conv2 = tf.keras.layers.Conv2D(
                filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
            )
            self.shortcut_conv = tf.keras.layers.Conv2D(
                filters, kernel_size=(1, 1), kernel_initializer="glorot_uniform", padding=pad
            )
        self.downsample_layer = tf.keras.layers.AvgPool2D((2, 2))
        self.downsample = downsample

    def call(self, x):
        print("")
        h1 = x
        h1 = tf.nn.relu(h1)
        h1 = self.conv1(h1)
        h1 = tf.nn.relu(h1)
        h1 = self.conv2(h1)
        if self.downsample:
            h1 = self.downsample_layer(h1)

        h2 = x
        h2 = self.shortcut_conv(h2)
        if self.downsample:
            h2 = self.downsample_layer(h2)

        return h1 + h2


class DiscOptResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, pad="same", spectral_norm=False):
        super(DiscOptResBlock, self).__init__(name="")

        if spectral_norm:
            self.conv1 = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
                )
            )
            self.conv2 = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
                )
            )
            self.shortcut_conv = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    kernel_initializer="glorot_uniform",
                    padding=pad,
                )
            )
        else:
            self.conv1 = tf.keras.layers.Conv2D(
                filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
            )
            self.conv2 = tf.keras.layers.Conv2D(
                filters, kernel_size, padding=pad, kernel_initializer="glorot_uniform",
            )
            self.shortcut_conv = tf.keras.layers.Conv2D(
                filters, kernel_size=(1, 1), kernel_initializer="glorot_uniform", padding=pad,
            )
        self.downsample_layer = tf.keras.layers.AvgPool2D((2, 2))

    def call(self, x):
        h1 = x
        h1 = self.conv1(h1)
        h1 = tf.nn.relu(h1)
        h1 = self.conv2(h1)
        h1 = self.downsample_layer(h1)

        h2 = x
        h2 = self.shortcut_conv(h2)
        h2 = self.downsample_layer(h2)

        return h1 + h2
