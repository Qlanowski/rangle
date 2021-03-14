#%%

import json
import cv2
import numpy as np
# ann_path = "ann/train_image.json"
# img_dir ="train_img"
ann_path = "ann/train_image.json"
img_dir ="train_img"
#%%
def load_dataset(img_dir, ann_file, count):
  with open(ann_file) as content:
    json_ann = json.load(content)

  anns = json_ann
  y = []
  y = np.array(list(map(lambda x: np.array(x['people'][0]['keypoints']).reshape(-1,3)[:,:2], anns )))
  for a in anns:
    keypoints = []
    for p in a['people']:
      r = np.array(p['keypoints']).reshape(-1,3)[:,:2]
      keypoints.append(r)
      break #TODO generating heatmaps for multiple people
    y.append(keypoints)
  sigma = np.array([0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107,0.089,0.107])*5
  sigma = tf.constant(sigma, dtype=tf.float32)
  h, v = generate_heatmaps(tf.constant(y), [224, 224],[56,56,23],sigma)


  return None, h
# %%

a,b =load_dataset(img_dir,ann_path,20)