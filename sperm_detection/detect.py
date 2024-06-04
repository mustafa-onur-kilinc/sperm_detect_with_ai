"""
Function (detect_video) for applying tracking to an input video.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2

from get_model import get_model
from utils.check_dir import get_next_run_directory
from tracker import Tracker


transform = transforms.Compose(
    [
        transforms.Resize((1080, 1920)),
        # Convert images to PyTorch tensors with values in [0, 1]
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]
)


def make_predictions(model, device, img):
    img_tensor = transform(img)
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
    
    # prediction[0]["boxes"], prediction[0]["labels"], 
    # prediction[0]["scores"]
    return prediction


def detect_video(model, device, video_path):
    tracker = Tracker()
    runs_dir = get_next_run_directory(r"sperm-detection/runs/detect")
    os.makedirs(runs_dir, exist_ok=True)
    output_video_path = os.path.join(runs_dir, "track_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (1920, 1080))
    i = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")

    frame_idx = 0
    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from video.")
            break

        frame_np = np.array(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        preds = make_predictions(model, device, pil_image)
        detections = []
        if preds:
            for pred in preds:
                # Filter predictions based on confidence score
                boxes = pred["boxes"]
                scores = pred["scores"]
                labels = pred["labels"].tolist()
                for box, score, label in zip(boxes, scores, labels):
                    # print(label, score)
                    if score > 0.5:
                        x_min, y_min, x_max, y_max = box.tolist()
                        
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        detections.append((x_center, y_center, width, 
                                           height, label))
                        cv2.rectangle(frame_np, (int(x_min)-3, int(y_min)-3), 
                                      (int(x_max)+3, int(y_max)+3), (0,0,255), 2)
                        
        tracker_predictions = tracker.update(
            detections
        )  # id, x, y, width, height, label
        
        if tracker_predictions:
            for pred in tracker_predictions:
                # Extract coordinates and dimensions
                # print(pred[1:3])
                x_center = int(pred[1])
                y_center = int(pred[2])
                width = int(pred[3])
                height = int(pred[4])
                label = int(pred[5])
                # Calculate rectangle coordinates
                x_min = x_center - width // 2
                y_min = y_center - height // 2
                x_max = x_center + width // 2
                y_max = y_center + height // 2

                color = (
                    (0, 255, 0) if label == 1 else (255, 0, 0)
                )  # Green for '0', Blue for '1' 

                # Draw the rectangle
                cv2.rectangle(frame_np, (x_min, y_min), (x_max, y_max), color, 2)
                # Put the tracker ID near the rectangle
                cv2.putText(
                    frame_np,
                    f"{pred[0]}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
        video_writer.write(frame_np)
        i += 1
        if i > 100:
            break
        print(f"frame : {frame_idx} processed")
        


if __name__ == "__main__":
    # Sample Usage
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else: 
        torch.device("cpu")
    model_name = "faster-rcnn"
    backbone_name = "resnet50"
    weights_path = r"weights_path"
    model = get_model(model_name, backbone_name, weights_path)
    model.eval()
    model.to(device)
    video_path = r"video_path"
    detect_video(model, device, video_path)
