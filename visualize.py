
# %%
import os
import sys
import glob
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE, NAMES
from nets.simple_baseline import SimpleBaseline
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import cost_functions as cf
from tfds_loader import load_ds, get_dataset_iterator
import utils.predictions as pu
import utils.plots as pl
# %%
dir_img ="train_img"
imgs = pl.load_images(dir_img,93)
original_imgs = pl.load_original_images(dir_img,93)

model = SimpleBaseline(INPUT_SHAPE)
model.load_weights(f'./models/all_tpu_simple.h5', by_name=True)
pred_hm = model.predict(imgs)

i=0
for img, org_img, hm in zip(imgs, original_imgs, pred_hm):
    i+=1
    if i>91:
        pl.plot_original_image(org_img, hm)
        pl.plot_image(img, hm)
