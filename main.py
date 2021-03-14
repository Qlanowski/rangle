#%%
import tensorflow as tf
# %%
# modele pobrane od gościa z tego posta
#https://github.com/qubvel/efficientnet/issues/104
ef = tf.keras.models.load_model('./EFLiteModels/L0.h5')

# %%
ef.summary()
#%%
# model = tf.keras.models.Model(inputs=ef.input, outputs=ef.get_layer('top_activation').output)
# model = tf.keras.models.Model(inputs=ef.input, outputs=ef.layers[-4].output)
# %%
# model.summary()
# %%
class MyModel(tf.keras.Model):

    def __init__(self, backbone_layers):
        super(MyModel, self).__init__()
        self.backbone = backbone_layers
        self.dense = tf.keras.layers.Dense(32, activation='relu', name="dense")
        self.head = tf.keras.layers.Dense(1,name="head")


    def call(self, x):
        for l in self.backbone:
          x = l(x)
        # x = self.dense(x)
        # return self.head(x)
        return x
# %%
model = MyModel(ef.layers[::-4])
#%%
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
#%%
X = tf.random.uniform(shape=[100,224,224,3])
Y = tf.random.uniform(shape=[100])
#%%
model.fit(x=X, y=Y, batch_size=10, epochs=15, verbose=1, validation_split=0.2)
# %%
X = tf.random.uniform(shape=[10,224,224,3])
Y = tf.random.uniform(shape=[10])

new_model = tf.keras.Sequential()
new_model.add(test)
new_model.add(tf.keras.layers.GlobalAveragePooling2D())
new_model.add(tf.keras.layers.Dense(100,activation='relu'))
new_model.add(tf.keras.layers.Dense(1,activation='relu'))
new_model.summary()

base_learning_rate = 0.0001
new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

new_model.fit(x=X, y=Y, batch_size=10, epochs=15, verbose=1, validation_split=0.2)