# %%
import tensorflow as tf


@tf.function
def mse(y_actual, y_pred):
    valid = tf.cast(tf.math.reduce_max(y_actual, axis=(1, 2)) > 0, dtype=tf.float32)
    valid_mask = tf.reshape(valid, [tf.shape(y_actual)[0], 1, 1, tf.shape(valid)[-1]])
    return tf.reduce_mean(tf.square(y_actual - y_pred) * valid_mask)


# %%
if __name__ == "__main__":

    a = tf.constant([
        [1, 2],
        [1, 2]
    ], dtype=tf.float32)
    b = tf.math.multiply(0.5, a)


# %%
