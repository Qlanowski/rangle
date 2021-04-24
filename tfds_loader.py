# %%
import tensorflow as tf
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE
from common import get_config
import utils.plots as pl
# %%


def get_dataset_iterator(dataset):
    return dataset.unbatch().as_numpy_iterator()


def generate_heatmap(kp, input_shape, output_shape):
    SIGMA = 2 * output_shape[0] / 64
    # SIGMA = 100
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.dtypes.cast(xx, tf.float32), (*output_shape[:2], 1))
    yy = tf.reshape(tf.dtypes.cast(yy, tf.float32), (*output_shape[:2], 1))

    x = tf.floor(tf.reshape(kp[:, 0], [-1, 1, output_shape[-1]])
                 / input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(tf.reshape(kp[:, 1], [-1, 1, output_shape[-1]])
                 / input_shape[0] * output_shape[0] + 0.5)

    heatmap = tf.exp(-(((xx - x) / SIGMA) ** 2) / 2 - (
        ((yy - y) / SIGMA) ** 2) / 2) * 255.

    valid = tf.cast(kp[:, -1] > 0, tf.float32)
    valid_mask = tf.reshape(valid, [1, valid.shape[-1]])
    return heatmap*valid_mask

def generate_heatmap_many_people(kp, input_shape, output_shape):
    SIGMA = 2 * output_shape[0] / 64
    # SIGMA = 100
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.dtypes.cast(xx, tf.float32), (*output_shape[:2], 1))
    yy = tf.reshape(tf.dtypes.cast(yy, tf.float32), (*output_shape[:2], 1))

    x = tf.floor(tf.reshape(kp[:, :, 0], [-1, 1, 1, output_shape[-1]])
                 / input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(tf.reshape(kp[:, :, 1], [-1, 1, 1, output_shape[-1]])
                 / input_shape[0] * output_shape[0] + 0.5)

    heatmap = tf.exp(-(((xx - x) / SIGMA) ** 2) / 2 - (
        ((yy - y) / SIGMA) ** 2) / 2) * 255.

    heatmap = tf.math.reduce_sum(heatmap, axis=0)

    kp_visible = tf.math.reduce_sum(kp, axis=0)
    valid = tf.cast(kp_visible[:, -1] > 0, tf.float32)
    valid_mask = tf.reshape(valid, [1, valid.shape[-1]])
    return heatmap * valid_mask


def prepro(img, height, width, kp, input_shape, output_shape):
    img = tf.cast(img, tf.float32)
    kp = tf.cast(kp, tf.float32)
    heatmap = generate_heatmap_many_people(kp, [height, width], output_shape)
    img = tf.image.resize(img, (input_shape[0], input_shape[1]))
    return img/255., heatmap


def assess_prepro(img_id, img, height, width, areas, keypoints, input_shape, output_shape):
    img, hm = prepro(img, height, width, keypoints, input_shape, output_shape)
    kp = tf.reshape(keypoints, [-1])
    kp = tf.pad(kp, [[0, tf.constant(12*69) - tf.size(kp)]])
    kp = tf.reshape(kp, [-1, 23, 3])
    areas = tf.pad(areas, [[0, tf.constant(12) - tf.size(areas)]])
    return img_id, img, height, width, areas, kp, hm


def parse_record(record):
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
        'areas': tf.io.VarLenFeature(dtype=tf.float32),
        'bboxes': tf.io.VarLenFeature(dtype=tf.float32),
        'keypoints': tf.io.VarLenFeature(dtype=tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    rec = tf.io.parse_single_example(record, feature)
    img = tf.image.decode_jpeg(rec['image'], channels=3)
    img_id = rec['image_id']
    height = tf.cast(rec['height'], dtype=tf.float32)
    width = tf.cast(rec['width'], dtype=tf.float32)
    areas = tf.sparse.to_dense(rec['areas'])
    bboxes = tf.reshape(tf.sparse.to_dense(rec['bboxes']), [-1, 4])
    keypoints = tf.reshape(tf.sparse.to_dense(rec['keypoints']), [-1, 23, 3])
    return img_id, img, height, width, areas, bboxes, keypoints

# def crop(img, height, width, keypoints, cfg):



def single_augmentation(img_id, img, height, width, areas, bboxes, keypoints, cfg):
    seed=(1,2)
    if cfg.DATASET.FLIP_PROB > 0 and tf.random.uniform([]) <= cfg.DATASET.FLIP_PROB:
        img = tf.image.flip_left_right(img)
        x = keypoints[:,:,0]
        x = tf.cast(width, dtype=tf.int64) - x
        x = tf.reshape(x, [-1, 23, 1])
        
        keypoints = tf.concat([x, keypoints[:,:,1:]], axis=-1)
        keypoints = tf.gather(keypoints, indices=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19], axis=1)

    if cfg.DATASET.CONTRAST_PROB > 0 and tf.random.uniform([]) <= cfg.DATASET.CONTRAST_PROB: 
        img = tf.image.stateless_random_contrast(img, lower=0.1, upper=0.9, seed=seed)

    if cfg.DATASET.HUE_PROB > 0 and tf.random.uniform([]) <= cfg.DATASET.HUE_PROB: 
        img = tf.image.stateless_random_hue(img, 0.2, seed)

    return img_id, img, height, width, areas, bboxes, keypoints


def load_ds(data_dir, batch_size, input_shape, output_shape, augmentation=False, shuffle=True, cfg=None):
    AUTO = tf.data.experimental.AUTOTUNE
    gcs_pattern = f'gs://rangle/tfrecords.zip/{data_dir}/*.tfrec'
    ds = tf.data.Dataset.list_files(gcs_pattern, shuffle=shuffle)

    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=10,
                       block_length=1,
                       num_parallel_calls=AUTO)
    ds = ds.map(lambda record: parse_record(record), num_parallel_calls=AUTO)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.repeat()

    if augmentation:
        ds = ds.map(lambda img_id, img, height, width, areas, bboxes, keypoints: single_augmentation(
            img_id, img, height, width, areas, bboxes, keypoints, cfg), num_parallel_calls=AUTO)

    ds = ds.map(lambda img_id, img, height, width, areas, bboxes, keypoints: prepro(
        img, height, width, keypoints, input_shape, output_shape), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size).prefetch(AUTO)

    return ds


def load_plain_ds(data_dir, batch_size, input_shape, output_shape):
    AUTO = tf.data.experimental.AUTOTUNE
    gcs_pattern = f'gs://rangle/tfrecords.zip/{data_dir}/*.tfrec'
    ds = tf.data.Dataset.list_files(gcs_pattern, shuffle=False)

    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=10,
                       block_length=1,
                       num_parallel_calls=AUTO)
    ds = ds.map(lambda record: parse_record(record), num_parallel_calls=AUTO)
    ds = ds.cache()
    ds = ds.map(lambda img_id, img, height, width, areas, bboxes, keypoints: assess_prepro(
        img_id, img, height, width, areas, keypoints, input_shape, output_shape))
    # ds = ds.map(lambda img_id, img, height, width, areas, bboxes, keypoints: prepro(img, height, width, keypoints, input_shape, output_shape))
    ds = ds.batch(batch_size).prefetch(AUTO)

    return ds


def to_image(img):
    img2 = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    img2[:, :, 0] = img  # same value in each channel
    img2[:, :, 1] = img
    img2[:, :, 2] = img
    return img2


if __name__ == "__main__":
    tf.random.set_seed(0)

    cfg = get_config()

    train_dataset = load_ds(cfg.DATASET.TRAIN_DIR, cfg.TRAIN.BATCH_SIZE,
                            cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE,
                            augmentation=True, shuffle=False, cfg=cfg)


    dataset = get_dataset_iterator(train_dataset)
    for pair in dataset:
        img = pair[0]
        hm = pair[1]
        pl.plot_image(img, hm)
    # train_dir = 'train'
    # val_dir = 'val'
    # batch_size = 32
    # ds = load_ds(val_dir, batch_size, INPUT_SHAPE, OUTPUT_SHAPE)

    # for i, (imgs, heatmaps) in enumerate(ds):
    #     for b in range(batch_size):
    #         img = imgs[b]
    #         heatmap = heatmaps[b]
    #         h = tf.image.resize(
    #             to_image(heatmap[:, :, i]), (img.shape[0], img.shape[1]))
    #         c = img + h*0.01
    #         imgplot = plt.imshow(c)
    #         plt.show()
    #         imgplot = plt.imshow(heatmap[:, :, i])
    #         plt.show()

# %%
