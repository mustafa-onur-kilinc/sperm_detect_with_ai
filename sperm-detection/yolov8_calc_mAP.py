"""
This script calculates mean Average Precision (mAP) for trained YOLOv8 
weight by using directory of dataset's train folder and test folder.

In order to work, this script needs a directory like this:
- main_dir
    - script_dir
        - yolov8_predict.py
    - config_dir
        - yaml_name
    - dataset_folder
        - train_folder
        - test_folder
    - result_dir[0]
        - result_dir[1]
    - runs_dir
        - detect_dir
            - weights_dir[0]
                - weights_dir[1]
                    - weights_name
"""

import os
import yaml
from ultralytics import YOLO
from tempfile import NamedTemporaryFile


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    main_dir = os.path.split(script_dir)[0]

    config_dirname = "config"
    yaml_name = "yolov8_calc_mAP.yaml"
    yaml_dir = os.path.join(main_dir, config_dirname, yaml_name)

    with open(yaml_dir, "r", encoding="utf8") as yaml_file:
        args_dict = yaml.safe_load(yaml_file)

    eval_on_test_dataset = args_dict["eval_on_test_dataset"]
    dataset_name = args_dict["dataset_name"]
    train_dirname = args_dict["train_dirname"]
    test_dirname = args_dict["test_dirname"]
    results_dirname = args_dict["results_dirname"]
    videos_count = args_dict["videos_count"]
    num_classes = args_dict["num_classes"]
    class_names = args_dict["class_names"]

    runs_dirname = args_dict["runs_dirname"]
    detect_dirname = args_dict["detect_dirname"]
    weights_dirname = args_dict["weights_dirname"]
    weights_name = args_dict["weights_name"]

    dataset_dir = os.path.join(main_dir, dataset_name)
    train_dir = os.path.join(dataset_dir, train_dirname)
    test_dir = os.path.join(dataset_dir, test_dirname)

    results_dir = os.path.join(main_dir, results_dirname[0], 
                               results_dirname[1])

    detect_dir = os.path.join(main_dir, runs_dirname, detect_dirname)
    weight_dir = os.path.join(detect_dir, weights_dirname[0], 
                              weights_dirname[1], weights_name)

    if eval_on_test_dataset:
        data_config = {
            # Not important
            "train": train_dir,
            # First eval on test dataset
            "val": test_dir,
            "nc": num_classes,
            "names": class_names,
        }
        results_path = os.path.join(results_dir, "results-yolo.txt")
    
        yaml_content = yaml.dump(data_config)
    
        yaml_write_path = "temp.yaml"
        # Write this YAML string to a temporary file
        with open(yaml_write_path, "w") as tmp:
            tmp.write(yaml_content)
    
        model = YOLO(weight_dir)
    
        metrics = model.val(data=yaml_write_path, save_json=True)
    
        # Write results under test dataset folder
        with open(results_path, "w") as f:
            f.write(f"mAP@0.5:0.95: {str(metrics.box.map)}")
            f.write("\n")
            f.write(f"mAP@0.5: {str(metrics.box.map50)}")
            f.write("\n")
            f.write(f"mAP@0.75: {str(metrics.box.map75)}")
            f.write("\n")
    else:
        for i in range(1, videos_count + 1):
            data_config = {
                # Not important
                "train": train_dir,
                # Iterate each video
                "val": os.path.join(results_dir, f"vid_{i}"),
                "nc": num_classes,
                "names": class_names,
            }
            results_path = (
                os.path.join(results_dir, f"vid_{i}", "results-yolo-2.txt")
            )
    
            yaml_content = yaml.dump(data_config)
    
            yaml_write_path = "temp.yaml"
            # Write this YAML string to a temporary file
            with open(yaml_write_path, "w") as tmp:
                tmp.write(yaml_content)
    
            model = YOLO(weight_dir)
    
            metrics = model.val(data=yaml_write_path, save_json=True)

            # Write results under video folder
            with open(results_path, "w") as f:
                f.write(f"mAP@0.5:0.95: {str(metrics.box.map)}")
                f.write("\n")
                f.write(f"mAP@0.5: {str(metrics.box.map50)}")
                f.write("\n")
                f.write(f"mAP@0.75: {str(metrics.box.map75)}")
                f.write("\n")
