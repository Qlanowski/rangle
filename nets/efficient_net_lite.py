import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

from .simple_baseline import add_regularization

def EfficientNetLite(size):
    regularizer = l2(1e-5)

    backbone = tf.keras.models.load_model(f'EFLiteModels/L{size}.h5')
    backbone = add_regularization(backbone, regularizer)

    x = backbone.layers[-4].output
    for i in range(3):
        x = layers.Conv2DTranspose(
            256,
            4,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizer,
            name='head_conv{}'.format(i + 1))(x)
        x = layers.BatchNormalization(name='head_bn{}'.format(i + 1))(x)
        x = layers.Activation('relu', name='head_act{}'.format(i + 1))(x)
    x = layers.Conv2D(
        23, # number of joints
        1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizer,
        name='final_conv')(x)
    return Model(backbone.input, x, name='sb_{}'.format('Resnet50'))