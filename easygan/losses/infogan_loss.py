import tensorflow as tf


def auxillary_loss(disc_target, disc_preds, cont_trg, cont_pred):
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=disc_target, logits=disc_preds))
    cont_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(cont_trg - cont_pred), axis=1))
    aux_loss = disc_loss + cont_loss
    return aux_loss
