#%%
import os
import sys
import glob
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import pickle

from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE
from utils.dictToObject import DictToObject
import load_data as ld
from nets.simple_baseline import SimpleBaseline
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import cost_functions as cf
from tfds_loader import load_ds

def create_model(cfg, spe):
  if cfg.MODEL.NAME == 'SimpleBaseline':
    model = SimpleBaseline(cfg.DATASET.INPUT_SHAPE)
  
  lr = cfg.TRAIN.LR * cfg.TRAIN.BATCH_SIZE / 32

  lr_schedule = WarmupCosineDecay(
              initial_learning_rate=lr,
              decay_steps=cfg.TRAIN.EPOCHS * spe,
              warmup_steps=cfg.TRAIN.WARMUP_EPOCHS * spe,
              warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
              
  model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cf.mse)
  return model

# %%
if len(sys.argv) > 1:
    cfg = DictToObject(yaml.safe_load(open(sys.argv[1])))
else:
    cfg =  DictToObject(yaml.safe_load(open("./configs/local.yaml")))


train_dataset = load_ds(cfg.DATASET.TRAIN_DIR, cfg.TRAIN.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
val_dataset = load_ds(cfg.DATASET.VAL_DIR, cfg.VAL.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)

spe = int(np.ceil(cfg.DATASET.TRAIN_SIZE / cfg.TRAIN.BATCH_SIZE))
val_spe = int(np.ceil(cfg.DATASET.VAL_SIZE / cfg.VAL.BATCH_SIZE))
if cfg.TPU:
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))

  strategy = tf.distribute.TPUStrategy(resolver)
  with strategy.scope():
    model = create_model(cfg, spe)
else:
  model = create_model(cfg, spe)
# %%
model.summary()
# %%
history = model.fit(train_dataset, epochs=cfg.TRAIN.EPOCHS, verbose=1, validation_data=val_dataset, validation_steps=val_spe, steps_per_epoch=spe)
# %%
with open(f'./models/{cfg.MODEL.SAVE_NAME}.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

weights_path = f'./models/{cfg.MODEL.SAVE_NAME}.h5'
model.save_weights(weights_path)

#%%
# model.load_weights(f'./models/all_tpu_simple.h5', by_name=True)
# y_pred = model.predict(X_train)
# #%%
# def visualize(x, y, pred):
#   j=7
#   plt.imshow(x)
#   plt.show()
#   plt.imshow(y[:,:,j])
#   plt.show()
#   plt.imshow(pred[:,:,j])
#   plt.show()
# i=4
# visualize(X_train[i], y_train[i], y_pred[i])


# %%
