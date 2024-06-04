"""
Script for extracting multi-frame annotations (original annotations of dataset we used)
to frame-by frame json annotations under an output folder
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
        desc=f"Extracting annotations from {json_path}...",
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
    # Sample usage
    json_multiframe_annotations_path = "labels.json" 
    output_path = "output_folder"
    extract_annotations(json_multiframe_annotations_path, output_path)
    

