"""
Script by Özgün Zeki BOZKURT
"""

import os
from tqdm import tqdm


def yolo_to_corner_coordinates(labels_dir, output_labels_dir, img_size):
    img_width, img_height = img_size
    print("Image shape: ", img_size)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    label_files = os.listdir(labels_dir)
    print("Processing", len(label_files), "label files...")

    for label_file in tqdm(label_files, desc="Converting labels"):
        label_file_path = os.path.join(labels_dir, label_file)
        output_file_path = os.path.join(output_labels_dir, label_file)
        converted_labels = []
        with open(label_file_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_id = int(parts[0]) + 1
                x_center, y_center, width, height = map(float, parts[1:])

                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height

                converted_labels.append(f"{class_id} {x1} {y1} {x2} {y2}")

        with open(output_file_path, "w") as output_file:
            for label in converted_labels:
                output_file.write(label + "\n")

    print("Conversion complete. Converted labels saved to:", output_labels_dir)


def origin_to_yolo_coordinates(labels_dir, output_labels_dir):
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    label_files = os.listdir(labels_dir)
    print("Processing", len(label_files), "label files...")

    for label_file in tqdm(
        label_files, desc="Converting labels"
    ):  # tqdm is optional for a progress bar
        label_file_path = os.path.join(labels_dir, label_file)
        output_file_path = os.path.join(output_labels_dir, label_file)
        converted_labels = []
        with open(label_file_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])  # Assuming class_id does not need adjustment
                x_corner, y_corner, width, height = map(float, parts[1:])

                # Calculate center x, center y, width, and height in normalized coordinates
                x_center = (x_corner + width / 2) / 100
                y_center = (y_corner + height / 2) / 100
                norm_width = width / 100
                norm_height = height / 100

                converted_labels.append(
                    f"{class_id} {x_center} {y_center} {norm_width} {norm_height}"
                )

        with open(output_file_path, "w") as output_file:
            for label in converted_labels:
                output_file.write(label + "\n")

    print("Conversion complete. Converted labels saved to:", output_labels_dir)


if __name__ == "__main__":

    labels_dir = f"labels-original"
    output_dir = f"labels-yolo"
    img_size = (1920, 1080)

    origin_to_yolo_coordinates(labels_dir, output_dir)

    labels_dir = f"labels-yolo"
    output_dir = f"labels-corner-coordinates"

    yolo_to_corner_coordinates(labels_dir, output_dir, img_size)
