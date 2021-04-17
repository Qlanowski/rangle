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
import argparse

from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE
from utils.dictToObject import DictToObject
import load_data as ld
from common import create_model, get_config, get_strategy
from lr_schedules import WarmupPiecewise
from tfds_loader import load_ds

def display_training_curves(training, validation, title):
  fig, ax = plt.subplots()
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])


cfg = get_strategy()

strategy = get_strategy(cfg.TPU)

train_dataset = load_ds(cfg.DATASET.TRAIN_DIR, cfg.TRAIN.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
val_dataset = load_ds(cfg.DATASET.VAL_DIR, cfg.VAL.BATCH_SIZE, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)


if strategy != None:
  with strategy.scope():
    model = create_model(cfg)
else:
  model = create_model(cfg)

model.summary()

history = model.fit(train_dataset, epochs=cfg.TRAIN.EPOCHS, verbose=1, validation_data=val_dataset, validation_steps=cfg.VAL_SPE, steps_per_epoch=cfg.SPE)

with open(f'./models/{cfg.MODEL.SAVE_NAME}.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

weights_path = f'./models/{cfg.MODEL.SAVE_NAME}.h5'
model.save_weights(weights_path)

#%%
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')

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
