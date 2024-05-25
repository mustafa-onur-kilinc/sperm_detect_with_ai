"""
This script trains YOLOv8 model or performs prediction with a trained 
YOLOv8 model on a custom dataset. 

In order to work, this script needs a directory like this:
- main_dir
    - script_dir
        - yolov8_train_predict-1.py
    - config_dir
        - yolov8_train_predict-1.yaml
    - dataset_folder
        - test_dir
            - images
            - labels
        - train_dir
            - images
            - labels
        - valid_dir
            - images
            - labels
        - dataset.yaml
    - runs_dir
        - detect_dir
            - train_dir
                - weights_dir
                    - best.pt
                    - last.pt
    - yolov8m.pt
    - yolov8s.pt
    - yolov8n.pt

Resources used to write this script:
Glenn Jocher, Burhan Qaddoumi, Laughing-q 2024,
"Model Training with Ultralytics YOLO", 
Ultralytics Inc., accessed 29 March 2024, 
<https://docs.ultralytics.com/modes/train/#train-settings>

ChengKang Tan 2023, "Custom DataSet in YOLO V8 !", A Medium Corporation,
accessed 29 March 2024, 
<https://medium.com/@hichengkang/custom-dataset-in-yolo-v8-bf7468313026>

Glenn Jocher 2023, 
"Non-Square images (Aspect ratio 16:9) for Training the YoloV8 network", 
GitHub, Inc., accessed 29 March 2024, 
<https://github.com/ultralytics/ultralytics/issues/3815#issuecomment-1641996238>

Glenn Jocher 2023, 
"Can the yolov8 support to train the non-square 
(the default input size is w == h) image?",
GitHub, Inc., accessed 29 March 2024,
<https://github.com/ultralytics/ultralytics/issues/1284#issuecomment-1780328109>

Glenn Jocher 2024, "Early-Stopping Metric", GitHub, Inc., 
accessed 29 March 2024,
<https://github.com/ultralytics/ultralytics/issues/4521#issuecomment-1888049643>

numpydoc maintainers n.d., "Style guide", accessed 28 April 2024, 
<https://numpydoc.readthedocs.io/en/latest/format.html>

Seting up custom config.yaml 2024, GitHub, Inc., accessed 28 April 2024, 
<https://github.com/ultralytics/ultralytics/issues/7715>

The idea to use markdown bullet points to display directory structure is
from here:
Md Ehsanul Haque 2023, "Guide: Generate a Markdown Representation of a 
Directory Structure on Windows", GitHub, Inc., accessed 29 April 2024,
<https://gist.github.com/EhsanulHaqueSiam/ceda13af0da9589d2f43fdae4ad6fdb1>
"""

import os
import yaml

from ultralytics import YOLO
from torch import cuda

class YOLOTrainAndPredict():
    """
    Defines training and prediction functions for YOLOv8 models.
    """
    
    def __init__(self) -> None:
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]

        config_dirname = "config"
        yaml_name = "yolov8_train_predict-1.yaml"
        yaml_dir = os.path.join(main_dir, config_dirname, yaml_name)

        with open(yaml_dir, "r", encoding="utf8") as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        self.is_training = args_dict["is_training"]
        self.resume_training = args_dict["resume_training"]
        
        train_dataset_name = args_dict["train_dataset_name"]
        dataset_yaml_name = args_dict["dataset_yaml_name"]
        custom_cfg_name = args_dict["custom_cfg_name"]
        
        test_dataset_name = args_dict["test_dataset_name"]
        patient_name = args_dict["patient_name"]
        is_patient_dir_telefon = args_dict["is_patient_dir_telefon"]
        is_test_dir_video = args_dict["is_test_dir_video"]
        test_video_name = args_dict["test_video_name"]

        runs_dirname = args_dict["runs_dirname"]
        train_folder_name = args_dict["train_folder_name"]
        self.weights_folder_name = args_dict["weights_folder_name"]
        self.yolo_weight = args_dict["yolo_weight"]

        self.custom_cfg_dir = os.path.join(config_dirname, custom_cfg_name)

        # YOLO Predict function arguments
        self.imgsz = args_dict["imgsz"]
        self.save = args_dict["save"]
        self.stream = args_dict["stream"]
        self.line_width = args_dict["line_width"]
        
        if is_patient_dir_telefon:
            patient_name.append("Telefon")
        else:
            patient_name.append("Ekran")

        if is_test_dir_video:
            test_video_name = test_video_name + ".mp4"

        train_dataset_dir = os.path.join(main_dir, train_dataset_name)
        self.yaml_dir = os.path.join(train_dataset_dir, dataset_yaml_name)
        
        test_dataset_dir = os.path.join(main_dir, test_dataset_name) 
        test_dir = os.path.join(test_dataset_dir, patient_name[0], 
                                patient_name[1])

        # To predict videos with YOLO
        self.test_video_dir = os.path.join(test_dir, test_video_name)

        self.runs_dir = os.path.join(main_dir, runs_dirname[0], 
                                     runs_dirname[1])
        self.train_dir = os.path.join(self.runs_dir, train_folder_name)

        self.device = 0 if cuda.is_available else 'cpu'

    def yolo_train(self, resume_training: bool = False):
        """
        Performs training according to parameters set in 
        self.custom_cfg_dir.

        Parameters
        ----------
        resume_training : bool, default=False
            Determines whether to start training from scratch or resume 
            training using weights from last_weights_dir variable. 
            True means resume training with last weights, False means 
            start training from scratch. Defaults to False
        
        Returns
        ----------
        None
        """

        # last.pt in case training stops unexpectedly
        last_weights_dir = os.path.join(self.train_dir, 
                                        self.weights_folder_name, "last.pt")
        
        if resume_training:
            model = YOLO(last_weights_dir)
            model.train(resume=True)
        else:
            model = YOLO(self.yolo_weight)
            model.train(data=self.yaml_dir, cfg=self.custom_cfg_dir, 
                        device=self.device, save_dir=self.runs_dir)
            
    def yolo_predict(self):
        """
        Performs prediction according to parameters set in __init__ 
        function.

        Parameters
        ----------
        None

        Returns
        ----------
        None        
        """

        best_weights_dir = os.path.join(self.train_dir, 
                                        self.weights_folder_name, "best.pt")
        
        model = YOLO(best_weights_dir)
        print(f"self.test_video_dir = {self.test_video_dir}")
        model.predict(source=self.test_video_dir, save=self.save, 
                      imgsz=self.imgsz, stream=self.stream, 
                      line_width=self.line_width)
        
    def train_or_predict(self):
        """
        Calls self.yolo_train() or self.yolo_predict() function 
        according to self.is_training's value

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        if self.is_training:
            self.yolo_train(resume_training=self.resume_training)
        else:
            self.yolo_predict()

if __name__ == "__main__":
    yoloTrainer = YOLOTrainAndPredict()

    yoloTrainer.train_or_predict()
    