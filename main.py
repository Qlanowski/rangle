#%%
import os
import sys
import glob
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from const import SIGMA, INPUT_SHAPE,OUTPUT_SHAPE
import load_data as ld
from nets.simple_baseline import SimpleBaseline
# %%
if len(sys.argv) - 1 > 0:
  model_name = sys.argv[0]
else:
  model_name = "default_name"
print(model_name)
train_dir =  'train_img'
train_ann = "ann/train_image.json"
train_count = 50

val_dir =  'val_img'
val_ann = "ann/val_image.json"
val_count = 10

X_train, anns = ld.load_dataset(val_dir, val_ann, val_count)
y_train = ld.generate_heatmaps(anns, OUTPUT_SHAPE)

# %%
model = SimpleBaseline(INPUT_SHAPE)

# %%
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.MSE,
              metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=32, epochs=15, verbose=1, validation_split=0.2)
# %%
weights_path = f'./models/{model_name}.h5'
model.save_weights(weights_path)
