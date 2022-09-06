from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from model.shared import normalize, denormalize, pixel_shuffle


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """The model architecture of EDSR."""
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")

def edsr_weightnorm(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """The model architecture of EDSR."""
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block_weightnorm(b, num_filters, res_block_scaling)
    b = conv2d_weightnorm(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    
    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="modified-edsr")

def res_block(x_in, filters, scaling):
    """The architecture of a residual block in EDSR (no batch-normalisation layers)."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def res_block_weightnorm(x_in, filters, scaling):
    """The architecture of a residual block in EDSR (no batch-normalisation layers)."""
    x = conv2d_weightnorm(filters, 3, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    """Sub-pixel convolution (combination of a convolution and a 'pixel shuffle' operation)."""
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def upsample_weightnorm(x, scale, num_filters):
    """Sub-pixel convolution (combination of a convolution and a 'pixel shuffle' operation)."""
    def upsample_1(x, factor, **kwargs):
        x = conv2d_weightnorm(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
