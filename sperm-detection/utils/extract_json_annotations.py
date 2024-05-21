"""
Script by Özgün Zeki BOZKURT
"""

import cv2
from PIL import Image
import os
import json
from tqdm import tqdm


def extract_annotations(json_path, output_path):
    video_name = os.path.basename(json_path)
    video_name = os.path.splitext(video_name)[0]

    with open(json_path, "r") as file:
        data = json.load(file)

    for track in tqdm(
        data[0]["annotations"][0]["result"],
        desc=f"Extracting annotations from {json_path}..",
    ):
        # print(track.keys())
        # print(track["value"].keys())
        try:
            track_label = track["value"]["labels"][0]
            if track_label == "Sperm":
                class_id = 0
            if track_label == "NonSperm":
                class_id = 1

            for bbox in track["value"]["sequence"]:
                x = bbox["x"]
                y = bbox["y"]
                width = bbox["width"]
                height = bbox["height"]
                frame = bbox["frame"]

                txt_file_path = os.path.join(output_path, f"{video_name}_{frame}.txt")
                with open(txt_file_path, "a") as txt_file:
                    txt_file.write(f"{class_id} {x} {y} {width} {height}\n")
        except:
            print(track)


if __name__ == "__main__":
    os.makedirs("labels", exist_ok=True)
    output_path = f"labels-original"
    data_path = "Skelometrik"

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                json_path = os.path.join(folder_path, file)
                extract_annotations(json_path, output_path)

    # extract_annotations(json_path, output_path)

    """
    image_path = f"dataset-2\Ekran_1\Vid0177\Vid0177_1.png"
    image = cv2.imread(image_path)

    img_height, img_width = image.shape[:2]

    # Iterate through annotations and draw bounding boxes
    annos_path = f"labels\Vid0177_1.txt"

    with open(annos_path) as f:
        annotations = f.readlines()

    for annotation in annotations:
        class_label, rel_x, rel_y, rel_width, rel_height = map(
            float, annotation.split()
        )

        # Calculate absolute coordinates
        x = int(rel_x * img_width / 100)
        y = int(rel_y * img_height / 100)
        width = int(rel_width * img_width / 100)
        height = int(rel_height * img_height / 100)

        # Draw bounding box
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    # Display the image
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
