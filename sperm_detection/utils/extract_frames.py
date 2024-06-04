"""
Extracts frames from an input video.
"""

import cv2
import os


def extract_frames(input_video, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.basename(input_video)
    video_name = os.path.splitext(video_name)[0]

    print(f"Extracting frames from {input_video}..")

    for frame_number in range(total_frames):

        ret, frame = cap.read()

        if not ret:
            print(f"Error: could not read frame {frame_number+1}")
            continue

        frame_name = f"{video_name}_{frame_number+1}.jpg"
        frame_path = os.path.join(output_folder, frame_name)

        cv2.imwrite(frame_path, frame)

    cap.release()

    print("All frames extracted successfully.")


if __name__ == "__main__":
    # Sample usage
    output_path = f"/images_folder"
    video_path = "/sample_video.mp4"
    extract_frames(video_path, output_path)