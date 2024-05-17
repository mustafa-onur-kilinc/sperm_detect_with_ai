"""
This script takes predicted labels from YOLOv8's saved prediction labels, 
takes ground truth labels from dataset folder and uses these files to 
calculate mAP and mAP50-95 values.

In order to work properly, this script needs a folder structure like 
this:

- main_directory
    - config_folder
        - yaml_name
    - helpers_folder
        - calculate_mAP_yolo.py
    - *dataset_folder
        - *patient_name
            - Ekran
                - 'video_name'_labels
                    - 'video_name'_1.txt
                    - 'video_name'_2.txt
                    - 'video_name'_3.txt
                    - (...)
            - Telefon
                - 'video_name'_labels
                    - 'video_name'_1.txt
                    - 'video_name'_2.txt
                    - 'video_name'_3.txt
                    - (...)
    - *runs
        - *detect
            - *predict_folder
                - 'video_name'_labels
                    - 'video_name'_1.txt
                    - 'video_name'_2.txt
                    - 'video_name'_3.txt
                    - (...)

Note: The names with asterisk sign (*) in front of them can have 
different names

Note_2: While 'video_name' can be changed, label folders are expected to
be in 'video_name'_labels format

Original script from Özgün Zeki BOZKURT, received on 26th March 2024.

Resources used to write this script:
nikhilaggarwal3 2023, "Read a file line by line in Python", 
GeeksforGeeks, accessed 23 April 2024, 
<https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/>

Sergei Belousov, Moritz Spranger, Berk Ott 2024, "mean_average_precision",
GitHub Inc., accessed 23 April 2024, 
<https://github.com/bes-dev/mean_average_precision>
"""

import os
import yaml
import torch
import numpy as np

from collections import Counter
from mean_average_precision import MetricBuilder

class MeanAPCalculator:
    """
    This class gets directory of ground truth and prediction labels
    relative to script, formats them from YOLO label format to "x_min
    y_min x_max y_max class" format, calculates mean Average Precision
    (mAP) using formatted labels and prints resulting values to screen
    """
    
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]

        config_dirname = "config"
        yaml_name = "calculate_mAP_yolo.yaml"
        yaml_dir = os.path.join(config_dirname, yaml_name)

        with open(yaml_dir) as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        dataset_name = args_dict["dataset_name"]

        # List to append 'Ekran'/'Telefon' to patient name 
        ground_truth_src_name = args_dict["ground_truth_src_name"]
        is_src_name_telefon = args_dict["is_src_name_telefon"]
        self.video_name = args_dict["video_name"]

        runs_dir_name = args_dict["runs_dir_name"]
        detect_dir_name = args_dict["detect_dir_name"]
        predict_dir_name = args_dict["predict_dir_name"]

        # Image size (width x height) is used to transform YOLO labels
        self.image_size = args_dict["image_size"]  

        video_label_dir_name = self.video_name + "_" + "labels"

        # Appends 'Telefon' or 'Ekran' to reduce risk of typos and 
        # to make choosing easier
        if is_src_name_telefon:
            ground_truth_src_name.append("Telefon")
        else:
            ground_truth_src_name.append("Ekran")

        dataset_dir = os.path.join(main_dir, dataset_name)
        ground_truth_src_dir = os.path.join(dataset_dir, 
                                            ground_truth_src_name[0], 
                                            ground_truth_src_name[1])
        self.ground_truth_labels_dir = os.path.join(ground_truth_src_dir, 
                                                    video_label_dir_name)
        
        runs_dir = os.path.join(main_dir, runs_dir_name)
        detect_dir = os.path.join(runs_dir, detect_dir_name)
        predict_dir = os.path.join(detect_dir, predict_dir_name)
        self.predicted_labels_dir = os.path.join(predict_dir, 
                                                 video_label_dir_name)

        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=3
        )

    def format_yolo_labels(self, label_line: str, is_prediction: bool = False):
        """
        Formats a line of YOLO label from "class x_middle y_middle width
        height" format to "x_min y_min x_max y_max class" format. 
        Appends confidence score at the end of line if label line 
        belongs to a prediction

        Parameters
        ----------
        label_line : str
            A line of YOLO label to format
        is_prediction : bool, default=False
            Determines if confidence score should be appended to the end
            of formatted line. Default value is False

        Returns
        ----------
        labels_array : NDArray[Any] | tuple(int, int, int, int, int)
            Formatted line of label. Is an NDArray if is_prediction is 
            True, a tuple if is_prediction is False
        """
        
        elements = label_line.split()

        bb_width = int(float(elements[3]) * self.image_size[0])
        bb_height = int(float(elements[4]) * self.image_size[1])

        bb_middle_x = int(float(elements[1]) * self.image_size[0])
        bb_middle_y = int(float(elements[2]) * self.image_size[1])

        x_min = bb_middle_x - (bb_width // 2)
        y_min = bb_middle_y - (bb_height // 2)

        x_max = bb_middle_x + (bb_width // 2)
        y_max = bb_middle_y + (bb_height // 2)

        if is_prediction:
            labels_array = np.array([x_min, y_min, x_max, y_max, 
                                     int(elements[0])])
            labels_array = np.append(labels_array, float(elements[5]))
        else:
            labels_array = (x_min, y_min, x_max, y_max, int(elements[0]))

        return labels_array
    
    def read_label_files(self):
        """
        Reads ground truth and prediction labels from their respective
        directories, formats those labels with self.format_yolo_labels()
        function, returns resulting ground_truths and predictions lists

        Parameters
        ----------
        None

        Returns
        ----------
        predictions : NDArray[Any]
            Contains formatted prediction labels
        ground_truths : NDArray[Any]
            Contains formatted ground truth labels
        """
        
        ground_truths = []
        predictions = []

        ground_truth_labels_count = len(os.listdir(self.ground_truth_labels_dir))
        prediction_labels_count = len(os.listdir(self.predicted_labels_dir))

        for label_num in range(1, ground_truth_labels_count + 1):
            ground_truth_filename = self.video_name + "_" + str(label_num) + ".txt"
            ground_truth_file_dir = os.path.join(self.ground_truth_labels_dir, 
                                                 ground_truth_filename)
            
            ground_truth_file = open(ground_truth_file_dir, mode='r')

            for line in ground_truth_file:
                x_min, y_min, x_max, y_max, label_id = self.format_yolo_labels(line)
                ground_truths.append(np.array([x_min, y_min, x_max, y_max, 
                                               label_id, 0, 0]))

            ground_truth_file.close()

        for label_num in range(1, prediction_labels_count + 1):
            prediction_filename = self.video_name + "_" + str(label_num) + ".txt"
            prediction_file_dir = os.path.join(self.predicted_labels_dir, 
                                               prediction_filename)
            
            prediction_file = open(prediction_file_dir, mode='r')

            for line in prediction_file:
                x_min, y_min, x_max, y_max, label_id, confidence = self.format_yolo_labels(line, is_prediction=True)
                predictions.append(np.array([x_min, y_min, x_max, y_max, 
                                             label_id, confidence]))

            prediction_file.close()

        return np.array(predictions), np.array(ground_truths)
    
    def calculate_mean_ap(self):
        """
        Gets ground truth and prediction labels from 
        self.read_label_files(), calculates mean Average Precision (mAP)
        using mean_average_precision module, prints resulting mAP\@0.5 
        and mAP\@0.5:0.05:0.95 values to screen

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        predictions, ground_truths = self.read_label_files()

        self.metric_fn.add(predictions, ground_truths)

        # Compute mAP at IoU threshold = 0.5
        voc_map_05 = self.metric_fn.value(iou_thresholds=0.5)["mAP"]
        print(f"mAP@0.5: {voc_map_05}")
    
        # Compute mAP for IoU thresholds from 0.5 to 0.95, 
        # in steps of 0.05
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        coco_map = self.metric_fn.value(iou_thresholds=iou_thresholds)["mAP"]
        print(f"mAP@0.5:0.05:0.95: {coco_map}")


if __name__ == "__main__":
    mAP_calculator = MeanAPCalculator()

    mAP_calculator.calculate_mean_ap()
