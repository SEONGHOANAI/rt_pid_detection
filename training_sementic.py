import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

json_file = "train.json"
with open(json_file, "r") as f:
    data = json.load(f)

lists = []

for i, image in enumerate(data["images"]):
    image_id = image["id"]
    image_path = image["file_name"]
    img_data = cv2.imread(image_path)
    print(image_path)
    img_label = np.zeros((img_data.shape[0], img_data.shape[1]), dtype=np.uint8)
    for annotation in data["annotations"]:
        if annotation["image_id"] == image_id:
            segmentation = np.array(annotation["segmentation"])
            segmentation = np.round(segmentation).astype(int)
            print(segmentation)
            cv2.fillPoly(img_label, [segmentation], (255, 255, 255))

    if np.any(img_label):
        cv2.imwrite(f"train_label/train_{i}.png", img_label)
        lists.append(i)
    else:
        print(f"No change in label for image {image_id}, not saving.")

print(lists)
