
# %%
import os
import sys
import glob
import json
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


from const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE, NAMES
import load_data as ld
from nets.simple_baseline import SimpleBaseline
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import cost_functions as cf
from tfds_loader import load_ds
import utils.predictions as pu
# %%


def to_image(img):
    img2 = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    img2[:, :, 0] = img  # same value in each channel
    img2[:, :, 1] = img
    img2[:, :, 2] = img
    return img2


train_dir = 'train'
val_dir = 'val'
batch_size = 32

# %%


def get_dataset_iterator(dataset):
    return dataset.unbatch().as_numpy_iterator()


# %%
train_dataset = load_ds(train_dir, batch_size,
                        INPUT_SHAPE, OUTPUT_SHAPE, False)
val_dataset = load_ds(val_dir, batch_size, INPUT_SHAPE,
                      OUTPUT_SHAPE, False, False)

val_iter = get_dataset_iterator(val_dataset)

model = SimpleBaseline(INPUT_SHAPE)
model.load_weights(f'./models/all_tpu_simple.h5', by_name=True)
# %%
def plot_heatmap(ax, img, hm, idx):
    ax.axis('off')
    h_img = tf.image.resize(to_image(hm), (img.shape[0], img.shape[1]))
    ax.imshow(img)
    ax.imshow(h_img[:, :, 0], cmap=plt.cm.viridis, alpha=0.5)
    ax.set_title(NAMES[idx], fontsize=12, color='black')

for i, (img, heatmap) in enumerate(val_iter):
    pred_hm = model.predict(tf.constant([img]))[0]
    pred_kp = pu.get_preds(pred_hm, img.shape)
    fig, axs = plt.subplots(3, 8, figsize=(15, 10))
    for j, ax in enumerate(axs.ravel()):
        j-=1
        if j==-1:
            ax.axis('off')
            ax.imshow(img, interpolation='bilinear')
            for k in range(pred_kp.shape[0]):
                ax.scatter(pred_kp[k][0], pred_kp[k][1], s=20, c='red', marker='o')
            ax.set_title("Predictions", fontsize=16, color='black')
        else:
            plot_heatmap(ax, img, pred_hm[:, :, j], j)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def display_one_flower(image, title, subplot, color):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=12, color=color)

# If model is provided, use it to generate predictions.


def display_nine_flowers(images, titles, title_colors=None):
    subplot = 331
    plt.figure(figsize=(13, 13))
    for i in range(9):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_flower(images[i], titles[i], 331+i, color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# for i, (img, heatmap) in enumerate(val_iter):
#     for j in range(23):
#         pred_hm = model.predict(tf.constant([img]))[0]
#         pred_kp = pu.get_preds(pred_hm, img.shape)
#         h = tf.image.resize(
#             to_image(pred_hm[:, :, j]), (img.shape[0], img.shape[1]))

#         im1 = plt.imshow(img, interpolation='bilinear')
#         plt.scatter(pred_kp[j][0], pred_kp[j][1], s=20, c='red', marker='o')
#         im2 = plt.imshow(h[:, :, 0], cmap=plt.cm.viridis,
#                          alpha=.5, interpolation='bilinear')
#         plt.show()
