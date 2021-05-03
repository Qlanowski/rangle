#%%
import json
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import os.path as osp

from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model
from common import get_basic_config
import utils.plots as pl
from evopose2d import EvoPose
from nets.evo_transfer import EvoPose2D_transfer

#%%
cfg = get_config()


model = tf.keras.models.load_model('./evo/evopose2d_XS_f32.h5', compile=False)
print(model.summary())
# if __name__ == '__main__':
#     cfg = get_basic_config('evopose2d_XS_f32')
#     model = EvoPose(cfg)
#     print(model.summary())
# %%
