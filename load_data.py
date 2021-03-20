# %%
import tensorflow as tf
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE


def load_dataset(img_dir, ann_file, count):
    with open(ann_file) as content:
        json_ann = json.load(content)

    anns = json_ann[:count]

    load_img = lambda path: tf.keras.preprocessing.image.load_img(
        path, grayscale=False, color_mode='rgb', target_size=(224, 224),
        interpolation='nearest'
    )
    i = 0
    imgs = []
    for ann in anns:
        i += 1
        name = str(ann['image_id'])
        file_name = f'{name.zfill(12)}.jpg'
        path = os.path.join(img_dir, file_name)
        imgs.append(np.asarray(load_img(path)))
    imgs = np.asarray(imgs)
    imgs = imgs/255
    return imgs.astype('float32'), anns


def generate_heatmap(kp, input_shape, output_shape):
    SIGMA = 2 * output_shape[0] / 64
    # SIGMA = 100
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.dtypes.cast(xx, tf.float32), (1, *output_shape[:2], 1))
    yy = tf.reshape(tf.dtypes.cast(yy, tf.float32), (1, *output_shape[:2], 1))

    x = tf.floor(tf.reshape(kp[:, 0], [-1, 1, output_shape[-1]])
                    / input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(tf.reshape(kp[:, 1], [-1, 1, output_shape[-1]])
                    / input_shape[0] * output_shape[0] + 0.5)

    heatmap = tf.exp(-(((xx - x) / SIGMA) ** 2) / 2 - (
                ((yy - y) / SIGMA) ** 2) / 2) * 255.

    valid = tf.cast(kp[:, -1] > 0, tf.float32)
    valid_mask = tf.reshape(valid, [heatmap.shape[0], 1, valid.shape[-1]])
    return heatmap*valid_mask


def generate_heatmaps(anns, output_shape):
    y = np.empty((len(anns), *output_shape), dtype=np.float32)
    for i, a in enumerate(anns):
        h = np.array([0])
        for p in a['people']:
            kp = np.array(p["keypoints"], dtype=np.float32).reshape(-1, 3)
            h = h + \
                generate_heatmap(kp, [a['height'], a['width']], output_shape)
        y[i] = h
    return y


# %%
if __name__ == "__main__":

    train_dir = 'train_img'
    train_ann = "ann/train_image.json"
    train_count = 10

    val_dir = 'val_img'
    val_ann = "ann/val_image.json"
    val_count = 300
# %%
    X_train, anns = load_dataset(train_dir, train_ann, train_count)
# %%
    y_train = generate_heatmaps(anns, OUTPUT_SHAPE)
# %%

    def visualize(x, y, pred):
        plt.imshow(x)
        plt.show()
        plt.imshow(y[:, :, 0])
        plt.show()
        plt.imshow(pred[:, :, 0])
        plt.show()
    i = 3
    visualize(X_train[i], y_train[i], y_train[i])




