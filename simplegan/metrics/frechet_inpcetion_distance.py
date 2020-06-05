import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
import numpy as np
from scipy import linalg

__all__ = ["fid"]


def fid(images1, images2):

    r"""
    Args:
        images1: a numpy array/tensor of images. Shape: NxHxWxC
        images2: a numpy array/tensor of images. Shape: NxHxWxC

    Return:
        Frechet inception distance between images
    """

    ## Taken from https://github.com/mseitzer/pytorch-fid/blob/011829daeccc84341c1e8e6061d10a640a495573/fid_score.py#L138
    def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    img1_shape = images1.shape
    if img1_shape[1] != 299:
        images1 = tf.image.resize(images1, size=(299, 299))

    img2_shape = images2.shape
    if img2_shape[1] != 299:
        images2 = tf.image.resize(images2, size=(299, 299))

    assert images1.shape[1:] == (299, 299, 3) and images2.shape[1:] == (
        299,
        299,
        3,
    ), "images must be of shape 299x299x3"

    inception = InceptionV3(weights="imagenet", include_top=False)
    preds1 = inception(images1)
    preds1 = layers.GlobalAveragePooling2D()(preds1)

    preds2 = inception(images2)
    preds2 = layers.GlobalAveragePooling2D()(preds2)

    mu1 = np.mean(preds1, axis=0)
    sigma1 = np.cov(preds1, rowvar=False)

    mu2 = np.mean(preds2, axis=0)
    sigma2 = np.cov(preds2, rowvar=False)

    fid = calculate_fid(mu1, sigma1, mu2, sigma2)

    return fid
