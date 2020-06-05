import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

__all__ = ["inception_score"]


def inception_score(images):

    img_shape = images.shape
    if img_shape[1] != 299:
        images = tf.image.resize(images, size=(299, 299))

    assert images.shape[1:] == (299, 299, 3), "images must be of shape 299x299x3"

    inception = InceptionV3(weights="imagenet")
    predictions = inception(images)

    in_scores = []
    mean_pred = tf.reduce_mean(predictions, axis=0)
    kl_div = tf.keras.losses.KLDivergence()

    for i in range(predictions.shape[0]):
        in_scores.append(kl_div(mean_pred, predictions[i, :]))

    return tf.math.exp(tf.reduce_mean(in_scores)).numpy()
