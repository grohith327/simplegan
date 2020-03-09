import tensorflow as tf

'''
Returns a cycle consistency loss and a identity loss

Paper: https://arxiv.org/abs/1703.10593

Code source: https://www.tensorflow.org/tutorials/generative/cyclegan#import_and_reuse_the_pix2pix_models
'''

__all__ = ['cycle_loss',
           'identity_loss']

def cycle_loss(real_img, cycle_img, LAMBDA):

    loss = tf.math.reduce_mean(tf.math.abs(real_img - cycle_img))
    loss *= LAMBDA
    return loss


def identity_loss(real_img, same_img, LAMBDA):

    loss = tf.reduce_mean(tf.math.abs(real_img - same_img))
    loss *= LAMBDA
    return loss * 0.5
