import tensorflow as tf
from ..layers.spectralnorm import SpectralNormalization

__all__ = ["SelfAttention"]


class SelfAttention(tf.keras.Model):
    def __init__(self, spectral_norm=True):
        super(SelfAttention, self).__init__()
        self.scaling_factor = tf.Variable(0.0)
        self.spectral_norm = spectral_norm

    def build(self, input):
        _, _, _, n_channels = input

        if self.spectral_norm:
            self.conv1x1_f = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_g = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_h = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 2, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

            self.conv1x1_attn = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
        else:
            self.conv1x1_f = tf.keras.layers.Conv2D(
                filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
            )
            self.conv1x1_g = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 8, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )
            self.conv1x1_h = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels // 2, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

            self.conv1x1_attn = SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=n_channels, kernel_size=(1, 1), padding="same", strides=(1, 1)
                )
            )

        self.g_maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.h_maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def call(self, x):
        batch_size, height, width, n_channels = x.shape
        f = self.conv1x1_f(x)
        f = tf.reshape(f, (batch_size, height * width, n_channels // 8))
        g = self.conv1x1_g(x)
        g = self.g_maxpool(g)
        g = tf.reshape(g, (batch_size, (height * width) // 4, n_channels // 8))
        attn_map = tf.matmul(f, g, transpose_b=True)
        attn_map = tf.nn.softmax(attn_map)

        h = self.conv1x1_h(x)
        h = self.h_maxpool(h)
        h = tf.reshape(h, (batch_size, (height * width) // 4, n_channels // 2))
        attn_h = tf.matmul(attn_map, h)
        attn_h = tf.reshape(attn_h, (batch_size, width, height, n_channels // 2))
        attn_h = self.conv1x1_attn(attn_h)

        out = x + (attn_h * self.scaling_factor)
        return out
