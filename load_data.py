#%%
import tensorflow as tf
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from const import SIGMA, INPUT_SHAPE,OUTPUT_SHAPE

def load_dataset(img_dir, ann_file, count):
    img_files = glob.glob(f"{img_dir}/*.jpg")[:count]
    ids = set(map(lambda f: int(os.path.split(f)[-1].split(".")[0]), img_files))

    with open(ann_file) as content:
        json_ann = json.load(content)

    anns = list(filter(lambda a:a['image_id'] in ids, json_ann))
    anns = sorted(anns, key=lambda a: a['image_id'])
    #   y = tf.constant(list(map(lambda x: np.array(x['people'][0]['keypoints']).reshape(-1,3), anns )),dtype=tf.float32)
    #   sigma = np.array([0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107])*5
    #   sigma = tf.constant(sigma, dtype=tf.float32)
    #   heatmaps = generate_heatmaps(tf.constant(y), [224, 224],[56,56,23],sigma)


    load_img = lambda path: tf.keras.preprocessing.image.load_img(
        path, grayscale=False, color_mode='rgb', target_size=(224,224),
        interpolation='nearest'
    )
    i=0
    imgs = []
    print(len(img_files))
    for path in img_files:
        i+=1
        print(i)
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
                    /input_shape[0] * output_shape[0] + 0.5)

    heatmap = tf.exp(-(((xx - x) / SIGMA) ** 2) / 2 - (
                ((yy - y) / SIGMA) ** 2) / 2) * 255.
    
    valid = tf.cast(kp[:, -1] > 0, tf.float32)
    valid_mask = tf.reshape(valid,[heatmap.shape[0], 1, valid.shape[-1]])
    return heatmap*valid_mask
    

def generate_heatmaps(anns, output_shape):
    y = np.empty((len(anns), *output_shape), dtype=np.float32)
    for i, a in enumerate(anns):
        h = np.array([0])
        for p in a['people']:
            kp = np.array(p["keypoints"],dtype=np.float32).reshape(-1,3)
            h = h + generate_heatmap(kp, [a['height'], a['width']], output_shape)
        y[i] = h
    return y/255.

#%%
if __name__ == "__main__":
   
    train_dir =  'train_img'
    train_ann = "ann/train_image.json"
    train_count = 50

    val_dir =  'val_img'
    val_ann = "ann/val_image.json"
    val_count = 300
#%%
    x, anns = load_dataset(val_dir, val_ann, val_count)
#%%
    y = generate_heatmaps(anns, OUTPUT_SHAPE)
