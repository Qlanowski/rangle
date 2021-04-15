#%%
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import utils.predictions as pu
from nets.simple_baseline import SimpleBaseline
from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE, NAMES
#%%
def visualize(img, joints, valid):
    for i, v in enumerate(valid):
        if v == 1:  # occluded
            cv2.circle(img, tuple(joints[i]), 1, (0, 255, 0))
        elif v == 2:  # visible
            cv2.circle(img, tuple(joints[i]), 1, (0, 0, 255))
    return img

def to_image(img):
    img2 = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    img2[:, :, 0] = img  # same value in each channel
    img2[:, :, 1] = img
    img2[:, :, 2] = img
    return img2

def plot_heatmap(ax, img, hm, idx):
    ax.axis('off')
    h_img = tf.image.resize(to_image(hm), (img.shape[0], img.shape[1]))
    ax.imshow(img)
    ax.imshow(h_img[:, :, 0], cmap=plt.cm.viridis, alpha=0.5)
    ax.set_title(NAMES[idx], fontsize=12, color='black')

def plot_original_image(img, pred_hm):
    pred_kp = pu.get_preds(pred_hm, img.shape)
    plt.axis('off')
    plt.imshow(img, interpolation='bilinear')
    for k in range(pred_kp.shape[0]):
        plt.scatter(pred_kp[k][0], pred_kp[k][1], s=10, c='red', marker='o')

    plt.show()

def plot_image(img, pred_hm):
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

def load_images(dir_name,count=-1):
    if count==-1:
        files = os.listdir(dir_name)
    else:
        files = os.listdir(dir_name)[:count]
    load_img = lambda path: tf.keras.preprocessing.image.load_img(
        path, grayscale=False, color_mode='rgb', target_size=(224, 224),
        interpolation='nearest'
    )
    imgs = [np.array(load_img(os.path.join(dir_name, file))) for file in files]
    imgs = np.array(imgs)
    return imgs/255.

def load_original_images(dir_name,count=-1):
    if count==-1:
        files = os.listdir(dir_name)
    else:
        files = os.listdir(dir_name)[:count]
    imgs = [np.array(Image.open(os.path.join(dir_name, file))) for file in files]
    return imgs
    