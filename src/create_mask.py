import os
import json
import numpy as np
import cv2


json_dir = "../data/jsons"

output_dir = "../data/full_masks"

json_files = os.listdir(json_dir)

print("Creating masks...\n")

for json_name in json_files:

    json_path = os.path.join(json_dir, json_name)

    with open(json_path, "r") as file:

        data = json.load(file)

    height = data["imageHeight"]

    width = data["imageWidth"]

    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data["shapes"]:

        points = np.array(
            shape["points"],
            dtype=np.int32
        )

        cv2.fillPoly(mask, [points], 255)

    save_name = json_name.replace(".json", ".png")

    save_path = os.path.join(
        output_dir,
        save_name
    )

    cv2.imwrite(save_path, mask)

    print(f"Saved: {save_name}")

print("\nMask creation completed")