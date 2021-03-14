# %%
# Drawing points on images
import json
import cv2
# ann_path = "ann/val_image.json"
# img_dir ="val_img"
ann_path = "ann/train_image.json"
img_dir ="train_img"
#%%
with open(ann_path) as json_val_ann:
    images = json.load(json_val_ann)

def id_to_image(id):
    return str(id).zfill(12) + ".jpg"

for filename in os.listdir(img_dir):
    img_ann = [i for i in images if id_to_image(i["image_id"])==filename][0]
    img = cv2.imread(f"{img_dir}/{filename}")
    for person in img_ann["people"]:
        p = person["keypoints"]
        for i in range(int(len(p)/3)):
            s = 2
            cv2.rectangle(img, (p[i*3]-s, p[i*3+1]-s), (p[i*3]+s, p[i*3+1]+s), (255,0,0), 2)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (p[i*3]+s,p[i*3+1]+s)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(img,str(i+1), bottomLeftCornerOfText, 
                font, fontScale, fontColor, lineType)
    cv2.imshow('image',img)
    cv2.waitKey(0)

# %%
