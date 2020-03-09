import tensorflow as tf

'''
                        -- Not used anywhere --

possibly be removed
'''


'''
Computes KL loss + reconstruction loss

source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Reference: https://arxiv.org/abs/1312.6114
'''

__all__ = ['vae_loss']

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def vae_loss(values):

    y_true, y_pred = values[0], values[1]
    z_mean, z_var = values[2], values[3]
    org_dim = values[4]

    recon_loss = cross_entropy(y_true, y_pred)
    recon_loss *= org_dim

    kl_loss = 1 + z_var - \
        tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = tf.keras.backend.mean(recon_loss + kl_loss)
    return vae_loss
