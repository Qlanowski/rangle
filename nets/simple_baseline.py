import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = './models/tmp_weights.h5'
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

def SimpleBaseline(input_shape):
    regularizer = l2(1e-5)
    weights = None

    backbone = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape)

    backbone = add_regularization(backbone, regularizer)
    print("ok")
    x = backbone.output
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
    print("ok2")
    x = layers.Conv2D(
        23, # number of joints
        1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizer,
        name='final_conv')(x)
    print("ok3")
    return Model(backbone.input, x, name='sb_{}'.format('Resnet50'))