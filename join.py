#%%
import json
# foot_ann_path = "foot_annotations/foot_val.json"
# coco_ann_path = "annotations/person_keypoints_val2017.json"
# out_path = "ann/val_person.json"
# out_images_path = "ann/val_image.json"

foot_ann_path = "foot_annotations/foot_train.json"
coco_ann_path = "annotations/person_keypoints_train2017.json"
out_path = "ann/train_person.json"
out_images_path = "ann/train_image.json"


with open(foot_ann_path) as json_file:
    foot_val = json.load(json_file)

with open(coco_ann_path) as json_file:
    coco_val = json.load(json_file)

foot_ids =  set([ int(a['id']) for a in foot_val["annotations"]])
full_ann = [ an for an in coco_val["annotations"] if int(an['id']) in foot_ids]

#%%
val_out = []
for full in full_ann:
    foot = [ an["keypoints"] for an in foot_val["annotations"] if int(an['id']) == int(full['id'])][0]
    original = coco_val
    val_out.append({
        "image_id": full["image_id"],
        "id": full["id"],
        "area": full["area"],
        "bbox": full["bbox"],
        "iscrowd": full["iscrowd"],
        "keypoints": full["keypoints"] + foot
    })

with open(out_path, 'w') as fout:
    json.dump(val_out, fout)
# %%
images = []
image_ids = []
for a in val_out:
    if a["image_id"] in image_ids:
        continue
    image_ids.append(a["image_id"])
    people = [ {"id": an["id"],"area": an["area"],"bbox": an["bbox"], "iscrowd": an["iscrowd"],"keypoints": an["keypoints"] } for an in val_out if int(an['image_id']) == int(a['image_id'])]
    images.append({
        "image_id": a["image_id"],
        "people": people
    })

with open(out_images_path, 'w') as fout:
    json.dump(images, fout)

#%%
import os
from shutil import copyfile
all_dir ='train2017/train2017'
needed_dir ='train_img'
needed_ann ='ann/train_image.json'


with open(needed_ann) as json_file:
    ann = json.load(json_file)

needed_images = set([str(a["image_id"]).zfill(12)+'.jpg' for a in ann])

all_images = os.listdir(all_dir)
#%%
for i in all_images:
    if i in needed_images:
        copyfile(f"{all_dir}/{i}",f"{needed_dir}/{i}")