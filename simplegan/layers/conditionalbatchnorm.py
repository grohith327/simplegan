import tensorflow as tf

__all__ = ["ConditionalBatchNorm"]


class ConditionalBatchNorm(tf.keras.layers.Layer):
    r"""
    Paper: https://arxiv.org/abs/1610.07629
    Attributes:
        n_classes (int): Number of classes in the dataset.
        decay_rate (float): Momentum used to perform batch norm.
    """

    def __init__(self, n_classes, decay_rate=0.99):
        super(ConditionalBatchNorm, self).__init__()
        self.n_classes = n_classes
        self.decay_rate = decay_rate

    def build(self, input_shape):
        channels_shape = input_shape[-1:]
        self.params_shape = tf.TensorShape([self.n_classes]).concatenate(channels_shape)
        self.gamma = self.add_weight(shape=self.params_shape, initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=self.params_shape, initializer="zeros", name="beta")

        self.moving_params_shape = tf.TensorShape([1, 1, 1]).concatenate(channels_shape)
        self.moving_mean = self.add_weight(
            shape=self.moving_params_shape,
            initializer="zeros",
            trainable=False,
            name="moving_mean",
        )
        self.moving_var = self.add_weight(
            shape=self.moving_params_shape,
            initializer="ones",
            trainable=False,
            name="moving_var",
        )

    def call(self, inputs, labels, is_training=True):
        inputs_shape = tf.TensorShape(inputs.shape)
        beta = tf.gather(self.beta, labels)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = tf.gather(self.gamma, labels)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        variance_epsilon = 1e-5

        if is_training:
            mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
            self.moving_mean.assign(
                self.moving_mean * self.decay_rate + mean * (1 - self.decay_rate)
            )
            self.moving_var.assign(
                self.moving_var * self.decay_rate + variance * (1 - self.decay_rate)
            )
            outputs = tf.nn.batch_normalization(
                inputs, mean, variance, beta, gamma, variance_epsilon
            )
        else:
            outputs = tf.nn.batch_normalization(
                inputs, self.moving_mean, self.moving_var, beta, gamma, variance_epsilon
            )
        outputs.set_shape(inputs_shape)
        return outputs
