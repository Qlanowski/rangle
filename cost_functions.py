#%%
import tensorflow as tf

def mse(y_actual, y_pred):
    return tf.square(y_actual - y_pred)

#%%
if __name__ == "__main__":
   
    a = tf.constant([
        [1,2],
        [1,2]
    ],dtype=tf.float32)
    b = tf.math.multiply(0.5, a)


# %%
