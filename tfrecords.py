# %%
import tensorflow as tf
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.const import SIGMA, INPUT_SHAPE, OUTPUT_SHAPE

# %%
# out_dir ='train'
# img_dir = 'train_img'
# ann_file = "ann/train_image.json"

out_dir ='val'
img_dir = 'val_img'
ann_file = "ann/val_image.json"


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _floats_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=np.array(value, dtype=np.float32).reshape(-1)))

def _int64s_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(value, dtype=np.int32).reshape(-1)))


def serialize(image, height, width, image_id, areas, bboxes, keypoints):
    feature = {
        'image': _bytes_feature(image),
        'image_id': _int64s_feature([image_id]),
        'areas': _floats_feature(areas),
        'bboxes': _floats_feature(bboxes),
        'keypoints': _int64s_feature(keypoints),
        'height': _int64s_feature([height]),
        'width': _int64s_feature([width])
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with open(ann_file) as content:
    anns = json.load(content)

i = 0
imgs = []
for ann in anns:
    i += 1
    name = str(ann['image_id'])
    file_name = f'{name.zfill(12)}.jpg'
    path = os.path.join(img_dir, file_name)
    with open(path,'rb') as file:
        img = file.read()

    height = ann['height']
    width = ann['width']
    areas = list(map(lambda p: p['area'], ann['people']))
    bboxes = list(map(lambda p: p['bbox'], ann['people']))
    keypoints = list(map(lambda p: p['keypoints'], ann['people']))


    save_file = path = os.path.join(out_dir, f'{name.zfill(12)}.tfrec')
    with tf.io.TFRecordWriter(save_file) as writer:
        data = serialize(img, height, width, name,  areas, bboxes, keypoints)
        writer.write(data)
    if i % 100 == 0:
        print(f'{i}/{len(anns)}')