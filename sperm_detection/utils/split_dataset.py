"""
Script for splitting dataset for train-test
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Function to copy files from source to destination
def copy_files(file_list, src_dir, dst_dir, dataset):
    for file_name in tqdm(dataset, desc=f"Moving files from {src_dir} to {dst_dir}"):
        full_filename = (
            file_name + ".jpg" if src_dir.endswith("images") else file_name + ".txt"
        )
        shutil.copy(
            os.path.join(src_dir, full_filename), os.path.join(dst_dir, full_filename)
        )


def split_dataset(base_dir, train_size=0.8):
    images_dir = os.path.join(base_dir, "images")
    labels_dirs = {
        "labels-original": os.path.join(base_dir, "labels-original"),
        "labels-corner-coordinates": os.path.join(
            base_dir, "labels-corner-coordinates"
        ),
        "labels-yolo": os.path.join(base_dir, "labels-yolo"),
    }

    # Create train/test directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    for label_key in labels_dirs:
        os.makedirs(os.path.join(train_dir, label_key), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label_key), exist_ok=True)

    # Get a list of all image filenames without extension
    image_files = [
        os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".jpg")
    ]

    # Split the image filenames
    train_files, test_files = train_test_split(
        image_files, train_size=train_size, random_state=42
    )

    # Move images
    copy_files(image_files, images_dir, os.path.join(train_dir, "images"), train_files)
    copy_files(image_files, images_dir, os.path.join(test_dir, "images"), test_files)

    # Move labels for each type
    for label_key, label_dir in labels_dirs.items():
        copy_files(
            image_files, label_dir, os.path.join(train_dir, label_key), train_files
        )
        copy_files(
            image_files, label_dir, os.path.join(test_dir, label_key), test_files
        )


if __name__ == "__main__":
    dataset_dir = "dataset"
    split_dataset(dataset_dir)
