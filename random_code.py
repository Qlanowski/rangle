#%%
import json
import cv2
ann_path = "ann/val_person.json"
# img_dir ="val_img"
# ann_path = "annotations/person_keypoints_val2017.json"
#%%
with open(ann_path) as json_val_ann:
    anns = json.load(json_val_ann)
# %%
# print(1)
# num_of_people = [a["image_id"] for a in anns["annotations"] if a["num_keypoints"]>0]
# print(f'num_of_people {len(num_of_people)}')
# print(f'num_of_images {len(list(set(num_of_people)))}')
print(len(anns))
