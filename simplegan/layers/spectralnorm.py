import tensorflow as tf

__all__ = ["SpectralNormalization"]


class SpectralNormalization(tf.keras.layers.Wrapper):
    r"""
    Paper: https://arxiv.org/pdf/1802.05957.pdf
    Implementation based on https://github.com/tensorflow/addons/pull/1244

    Spectral norm is computed using power iterations.

    Attributes:
        layer (tf.keras.layer): Input layer to be taken spectral norm
        power_iternations (int): Number of power iterations to approximate the values"
    """

    def __init__(self, layer, power_iterations=5, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations

    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape).as_list()
        self.layer.build(input_shape)

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.layer.kernel.dtype,
        )

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.normalize_weights()
        output = self.layer(inputs)
        return output

    def normalize_weights(self):
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        for i in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
            u = tf.math.l2_normalize(tf.matmul(v, w))

        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))

        self.w.assign(self.w / sigma)
        self.u.assign(u)
