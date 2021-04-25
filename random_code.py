#%%
import json
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model

ann_path = "ann/train_image.json"
# img_dir ="val_img"
# ann_path = "annotations/person_keypoints_val2017.json"

#%%
def EfficientNet(size):
    regularizer = l2(1e-5)

    backbone = tf.keras.models.load_model(f'EFLiteModels/L{size}.h5')
    x = backbone.layers[-4].output
    for i in range(3):
        x = layers.Conv2DTranspose(
            256,
            4,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizer,
            name='head_conv{}'.format(i + 1))(x)
        x = layers.BatchNormalization(name='head_bn{}'.format(i + 1))(x)
        x = layers.Activation('relu', name='head_act{}'.format(i + 1))(x)

    x = layers.Conv2D(
        23, # number of joints
        1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizer,
        name='final_conv')(x)
    return Model(backbone.input, x, name=f'EfficientNet_Lite{size}')

#%%
model = EfficientNet(4)
model.summary()
#%%
with open(ann_path) as json_val_ann:
    anns = json.load(json_val_ann)
# %%
nums = [len(ann['people']) for ann in anns]
m = max(nums)
# print(1)
# num_of_people = [a["image_id"] for a in anns["annotations"] if a["num_keypoints"]>0]
# print(f'num_of_people {len(num_of_people)}')
# print(f'num_of_images {len(list(set(num_of_people)))}')
print(len(m))
