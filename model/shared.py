import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


# ---------------------------------------
#  Evaluation
# ---------------------------------------


def resolve_single(model, lr):
    """Super-resolves a single LR image using a given model."""
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    """Super-resolves a batch of LR images using a given model."""
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    """Evaluates model on a dataset in terms of PSNR and SSIM metrics."""
    psnr_values = []
    ssim_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        ssim_value = ssim(hr, sr)[0]
        ssim_values.append(ssim_value)
    return tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    """Pre-processes RGB images by subtracting the mean RGB value of DIV2K dataset 
    and then normalizes them to approximately [0, 1]."""
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    """Inverse of normalize."""
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    """Computes PSNR between HR and SR images."""
    return tf.image.psnr(x1, x2, max_val=255)

def ssim(x1, x2):
    """Computes SSIM between HR and SR images."""
    return tf.image.ssim(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    """Pixel shuffle operator."""
    return lambda x: tf.nn.depth_to_space(x, scale) 