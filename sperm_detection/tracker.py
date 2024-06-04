"""
Tracking module implementation
"""

import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

def bbox_from_center(x_center, y_center, width, height):
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

def iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class Tracker:
    def __init__(self):
        self.trackers = []
        self.track_ids = 0
        self.iteration = 0
        self.trace_length = 10
        self.penalty = 200

    def create_kalman_filter(self, x, y, vx=0, vy=0, ax=0, ay=0, dt=0.1):
        """
        Create a new Kalman Filter for a track including acceleration.
        - x, y: Initial position
        - vx, vy: Initial velocity (default is 0)
        - ax, ay: Initial acceleration (default is 0)
        - dt: Time step (delta_t)
        """
        
        kf = cv2.KalmanFilter(
            6, 2
        )  # 6 state variables (x, y, vx, vy, ax, ay), 2 measured variables (x, y)

        # State Transition Matrix (Model dynamics)
        # x = x0 + vx*dt + 0.5*ax*dt^2
        # vx = vx0 + ax*dt
        # ax = ax0 (constant acceleration model)
        kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement Matrix (Only position is measured)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32
        )

        # Example small noise
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 0.1
        )  # Low Measurement noise

        # Initial error covariance
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.05  

        # Initial State
        kf.statePost = np.array([x, y, vx, vy, ax, ay], dtype=np.float32)

        return kf

    def create_new_trackers(self, detections, unassigned_detections):
        for d_idx in unassigned_detections:
            det = detections[d_idx]
            self.trackers.append(
                {
                    "kf": self.create_kalman_filter(det[0], det[1]),
                    "id": self.track_ids,
                    "width": det[2],
                    "height": det[3],
                    "label": det[4],
                    "missed": 0,
                    "age" : 0,
                    "hitstreak": 0
                }
            )
            self.track_ids += 1

    def process_unassigned_trackers(self, unassigned_trackers):
        trackers_to_remove = []
        # Identify trackers that need to be removed
        for t_idx in list(unassigned_trackers):
            self.trackers[t_idx]["missed"] += 1
            self.trackers[t_idx]["hitstreak"] = 0
            if self.trackers[t_idx]["missed"] > self.trace_length:
                trackers_to_remove.append(t_idx)
        for t_idx in sorted(trackers_to_remove, reverse=True):
            self.trackers.pop(t_idx)

    def update(self, detections):
        self.iteration += 1
        predictions = []

        for tracker in self.trackers:
            prediction = tracker["kf"].predict()

            # Get [x, y] from [x, y, vx, vy]
            predictions.append(prediction[:2].flatten())

        # Calculate IoU-based cost matrix
        cost_matrix = np.zeros((len(self.trackers), len(detections)))
        for t_idx, tracker in enumerate(self.trackers):
            tracker_bbox = bbox_from_center(
                tracker["kf"].statePost[0],  # x_center
                tracker["kf"].statePost[1],  # y_center
                tracker["width"],
                tracker["height"]
            )
            for d_idx, detection in enumerate(detections):
                det_bbox = bbox_from_center(
                    detection[0],  # x_center
                    detection[1],  # y_center
                    detection[2],  # width
                    detection[3]   # height
                )
                # IoU cost is 1 - IoU because the Hungarian algorithm 
                # minimizes the cost
                iou_score = iou(tracker_bbox, det_bbox)
                cost_matrix[t_idx, d_idx] = 1 - iou_score

                # Reward consistent trackers 
                # (prevents temporary duplicate detections to override 
                # old tracker-detection with new tracker-detection pair)
                if tracker["age"] > 10 and iou_score > 0.3:
                    cost_matrix[t_idx, d_idx] -= 0.5
                

                

        # Solve the ID association problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Prevent tracker assignments to detections with distance 
        # more than threshold.
        assignments = []
        for r, c in zip(row_ind, col_ind):
            # Eliminate assignments with 0 IoU
            if cost_matrix[r,c] < 0.99:
                assignments.append((r, c))
            

        assigned_trackers = {r for r, _ in assignments}
        assigned_detections = {c for _, c in assignments}

        # Update assigned trackers
        for t_idx, d_idx in assignments:
            tracker = self.trackers[t_idx]
            detection = detections[d_idx]
            measurement = np.array([[detection[0]], [detection[1]]], np.float32)
            tracker["kf"].correct(measurement)
            tracker["width"] = detection[2]
            tracker["height"] = detection[3]
            tracker["label"] = detection[4]
            tracker["missed"] = 0
            tracker["age"] += 1
            tracker["hitstreak"] += 1

        # Handle unassigned trackers
        all_trackers = set(range(len(self.trackers)))
        unassigned_trackers = all_trackers - assigned_trackers
        self.process_unassigned_trackers(unassigned_trackers)

        # Create new trackers for unassigned detections
        all_detections = set(range(len(detections)))
        unassigned_detections = all_detections - assigned_detections
        self.create_new_trackers(detections, unassigned_detections)

        return [
            (
                tr["id"],
                tr["kf"].statePost[0],
                tr["kf"].statePost[1],
                tr["width"],
                tr["height"],
                tr["label"],
                tr["age"],
            )
            for tr in self.trackers
            if tr["missed"] <= 3 # Return trackers that is recently 
                                 # assigned to a detection.
        ]
