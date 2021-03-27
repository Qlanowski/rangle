#%%
import os
import sys
import glob
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


from utils.const import SIGMA, INPUT_SHAPE,OUTPUT_SHAPE
import load_data as ld
from nets.simple_baseline import SimpleBaseline
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import cost_functions as cf
from tfds_loader import load_ds
# %%
# if len(sys.argv) - 1 > 0:
#   model_name = sys.argv[0]
# else:
model_name = "200_simple"
print(model_name)

train_dir =  'train'
val_dir =  'val'
batch_size = 32



# X_train, anns = ld.load_dataset(val_dir, val_ann, val_count)
# y_train = ld.generate_heatmaps(anns, OUTPUT_SHAPE)
train_dataset = load_ds(train_dir, batch_size, INPUT_SHAPE, OUTPUT_SHAPE)
train_size = len(os.listdir(train_dir))
val_dataset = load_ds(val_dir, batch_size, INPUT_SHAPE, OUTPUT_SHAPE)
val_size = len(os.listdir(val_dir))

# %%
model = SimpleBaseline(INPUT_SHAPE)

# %%
WARMUP_EPOCHS = 5
WARMUP_FACTOR = 0.1
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.00025
spe = int(np.ceil(train_size / BATCH_SIZE))

lr = LR * BATCH_SIZE / 32
lr_schedule = WarmupCosineDecay(
            initial_learning_rate=lr,
            decay_steps=EPOCHS * spe,
            warmup_steps=WARMUP_EPOCHS * spe,
            warmup_factor=WARMUP_FACTOR)
            
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cf.mse)
model.fit(train_dataset, epochs=EPOCHS, verbose=1, validation_data=val_dataset, steps_per_epoch=spe)
# %%
weights_path = f'./models/{model_name}.h5'
model.save_weights(weights_path)

#%%
model.load_weights(f'./models/all_tpu_simple.h5', by_name=True)
y_pred = model.predict(X_train)
#%%
def visualize(x, y, pred):
  j=7
  plt.imshow(x)
  plt.show()
  plt.imshow(y[:,:,j])
  plt.show()
  plt.imshow(pred[:,:,j])
  plt.show()
i=4
visualize(X_train[i], y_train[i], y_pred[i])


# %%
