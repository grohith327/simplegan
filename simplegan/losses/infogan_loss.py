import tensorflow as tf


__all__ = ["auxillary_loss"]


def auxillary_loss(disc_target, disc_preds, cont_trg, cont_pred):

    r"""
    Args:
        disc_target (tensor): a tensor representing the discriminator target ouput
        disc_preds (tensor): a tensor representing the prediction of discriminator
        cont_trg (tensor): a tensor representing the content of the target
        cont_pred (tensor): a tensor representing the predicted content

    Return:
        a tensor representing the auxillary loss
    """

    disc_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_target, logits=disc_preds)
    )
    cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_trg - cont_pred), axis=1))
    aux_loss = disc_loss + cont_loss

    return aux_loss
