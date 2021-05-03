

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model


def EvoPose2D_transfer(size):
    regularizer = l2(1e-5)

    backbone = tf.keras.models.load_model(f'evo/evopose2d_{size}_f32.h5')

    x = backbone.layers[-2].output
    x = layers.Conv2D(
        23, # number of joints
        1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizer,
        name='final_conv')(x)
    return Model(backbone.input, x, name='sb_{}'.format(f'EvoPose2D_{size}_transfer'))