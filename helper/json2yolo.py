"""
This Python script is written to read a JSON file and transform 
annotations in that file to YOLO format.

In order to work, this script should be in the same folder as the folder 
of JSON files. For example:

- main_dir
    - helpers_folder
        - json2yolo.py
    - config_folder
        - yaml_name
    - dataset_folder
        - Patient_1
            - Ekran
                - json_filename
                - (...)
            - Telefon
                - json_filename
                - (...)
        - Patient_2
            - (...)

Resources used to write this script:
Tejashwi5 2023, "Read JSON file using Python", GeeksforGeeks,
accessed 26 March 2024, 
<https://www.geeksforgeeks.org/read-json-file-using-python/>

Samy Vilar 2012, 
"Difference between '{' and '[' when formatting JSON object",
Stack Exchange Inc., accessed 26 March 2024, 
<https://stackoverflow.com/a/11045748>

Jitender Mahlawat 2012, 
"Difference between '{' and '[' when formatting JSON object", 
Stack Exchange Inc., accessed 26 March 2024, 
<https://stackoverflow.com/a/11045675>

GeeksforGeeks 2023, "Reading and Writing to text files in Python", 
GeeksforGeeks, accessed 26 March 2024, 
<https://www.geeksforgeeks.org/reading-writing-text-files-python/>
"""

import os
import json
import yaml

class JSONToYOLO():
    """
    This class gets label files in JSON format from dataset directory,
    gets label information for every frame from JSON files, transforms 
    label information to YOLO format, saves YOLO formatted labels for 
    every frame to its own TXT file.
    """
    
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]

        config_dirname = "config"
        yaml_name = "json2yolo.yaml"
        yaml_dir = os.path.join(main_dir, config_dirname, yaml_name)

        with open(yaml_dir, "r", encoding="utf8") as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        self.frame_width = args_dict["frame_width"]
        self.frame_height = args_dict["frame_height"]

        dataset_name = args_dict["dataset_name"]

        src_name = args_dict["src_name"]

        is_src_name_telefon = args_dict["is_src_name_telefon"]  

        # Using if-else block to make choosing Telefon/Ekran less prone
        # to error and easier
        if is_src_name_telefon:
            src_name.append("Telefon")
        else:
            src_name.append("Ekran")

        json_file_name = args_dict["json_file_name"]

        dataset_dir = os.path.join(main_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"{dataset_dir} isn't an existing folder!!!")
            exit()
        
        json_folder_dir = os.path.join(dataset_dir, src_name[0], src_name[1])
        
        if not os.path.exists(json_folder_dir):
            print(f"{json_folder_dir} isn't an existing folder!!!")
            exit()
        
        self.json_file_dir = os.path.join(json_folder_dir, json_file_name)
        
        if not os.path.exists(self.json_file_dir):
            print(f"{self.json_file_dir} isn't an existing file!!!")
            exit()
        
        # json_file_name.split("_")[-1] gives "formatted.json" assuming 
        # json_file_name = videoName_formatted.json
        suffix_index = json_file_name.find(json_file_name.split("_")[-1])

        # suffix_index - 1 is to remove "_" before formatted.json
        labels_folder_name = json_file_name[:suffix_index - 1]
        self.video_name = labels_folder_name
        labels_folder_name = labels_folder_name + "_" + "labels"
        
        # labels_folder_dir is the directory to save labels of frames
        self.yolo_labels_dir = os.path.join(json_folder_dir, 
                                            labels_folder_name)
        
        try:
            if not os.path.exists(self.yolo_labels_dir):
                os.mkdir(self.yolo_labels_dir)
        except OSError as e:
            print(e)
            exit()
        
    def transform_labels_to_yolo(self):
        """
        Opens json file in self.json_file_dir, gets necessary label 
        information from json file, transforms it to YOLO format and 
        saves YOLO formatted labels to a TXT file in 
        self.yolo_labels_dir

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        json_file = open(self.json_file_dir)
        data = json.load(json_file)

        # result variable is used to increase readability
        results = data[0]['annotations'][0]['result']

        """
        data --> List (elements accessed with [i], i is an int)
        |--> [0]
        
        data[0] --> Dict (elements accessed with [key], key is a str)
        |--> "id"
        |--> "annotations"
        |--> "file_upload"
        |--> "drafts"
        | (...)
        
        data[0]['annotations'] --> List
        |--> [0]
        
        data[0]['annotations'][0] --> Dict
        |--> "id"
        |--> "completed_by"
        |--> "result"
        |--> "was_cancelled"
        |--> "ground_truth"
        |--> "created_at"
        | (...)
        
        data[0]['annotations'][0]['result'] --> List
        |--> [0]
        |--> [1]
        |--> [2]
        | (...)
        |--> [29]
        
        Note: data[0]['annotations'][0]['result'] will be referred to as 
        results from this point
        
        results[i] (0 <= i < sperm count(?)) --> Dict
        |--> "value"
        |--> "id"
        |--> "from_name"
        |--> "to_name"
        |--> "type"
        |--> "origin"
        
        results[i]['value'] (0 <= i < sperm count(?)) --> Dict
        |--> "framesCount"
        |--> "duration"
        |--> "sequence"
        |--> "labels"
        
        results[i]['value']['sequence'] (0 <= i < sperm count(?)) --> List
        |--> [0]
        |--> [1]
        |--> [2]
        |--> (...)
        |--> [a] (a = last sequence's ['frame'] - first sequence's 
                  ['frame'], a <= ['framesCount'] - 1)
        
        Note: results[i]['value']['sequence'] will be referred to as 
        sequences from this point
        
        Note: a = last sequence's ['frame'] - first sequence's ['frame'], 
        a <= ['framesCount'] - 1
        
        sequences[j] (0 <= j <= a) --> Dict
        |--> "frame"
        |--> "enabled"
        |--> "rotation"
        |--> "x"
        |--> "y"
        |--> "width"
        |--> "height"
        |--> "time"
        
        results[i]['value']['labels'] (0 <= i < 30) --> List
        |--> [0]
        
        results[i]['value']['labels'][0] --> String
        |--> Sperm
        """

        labels_list = []
        
        for result in results:
            labels = result['value']['labels']
        
            if labels[0] not in labels_list:
                labels_list.append(labels[0])
        
        """
        Let's assume x, y values in JSON file are found like this: 
        Let x_bb and y_bb show coordinates of bounding box's upper left corner, 
        let image width be 1920 px and image height 1080 px.
        
        x_json = (x_bb / 1920) * 100
        y_json = (y_bb / 1080) * 100
        
        In order to find x and y for YOLO format, we can find x_bb and y_bb, 
        add half of bounding box width and bounding box height to them, 
        divide resulting x_bb and y_bb by image width and image height 
        respectively.
        
        x_bb = (x_json / 100) * 1920
        y_bb = (y_json / 100) * 1080
        
        x_bb = x_bb + (w_bb / 2)
        y_bb = y_bb + (h_bb / 2)
        
        x_yolo = x_bb / 1920
        y_yolo = y_bb / 1080
        
        Let's assume width and height values in JSON file are found like this: 
        Let w_bb and h_bb show width and height of bounding box, 
        let image width be 1920 px and image height be 1080 px.
        
        width = (w_bb / 1920) * 100
        height = (h_bb / 1080) * 100
        
        In order to find bounding box width and heigth for YOLO format, 
        we can find w_bb and h_bb and divide them by image width and image 
        height respectively:
        
        w_bb = (width / 100) * 1920
        h_bb = (height / 100) * 1080
        
        w_yolo = w_bb / 1920
        h_yolo = h_bb / 1080
        
        w_yolo = [(width / 100) * 1920] / 1920 = (width / 100)
        h_yolo = [(height / 100) * 1080] / 1080 = (height / 100)
        """
        
        yolo_labels_list = []
        
        # Every value's framesCount seems to be the same
        frames_count = results[0]['value']['framesCount']
        
        for frame_count in range(1, frames_count + 1):
            labels_file_name = (self.video_name + "_" + str(frame_count) + 
                                ".txt")
            labels_file = open(os.path.join(self.yolo_labels_dir, 
                                            labels_file_name), "a")
        
            for result in results:
                sequences = result['value']['sequence']
                labels = result['value']['labels']
        
                for sequence in sequences:         
                    if sequence['frame'] == frame_count:
                        width_px = (sequence['width'] / 100 * 
                                    self.frame_width)
                        height_px = (sequence['height'] / 100 * 
                                     self.frame_height)
                        x_px = sequence['x'] / 100 * self.frame_width
                        y_px = sequence['y'] / 100 * self.frame_height
                        x_px = x_px + (width_px / 2)
                        y_px = y_px + (height_px / 2)
        
                        yolo_labels_list.append(str(labels_list.index(labels[0])))
                        yolo_labels_list.append(" ")
                        yolo_labels_list.append(str(x_px / self.frame_width))
                        yolo_labels_list.append(" ")
                        yolo_labels_list.append(str(y_px / self.frame_height))
                        yolo_labels_list.append(" ")
                        yolo_labels_list.append(str(sequence['width'] / 100))
                        yolo_labels_list.append(" ")
                        yolo_labels_list.append(str(sequence['height'] / 100))
                        yolo_labels_list.append("\n")
        
                        print(f"Writing to file {labels_file_name}...")
                        labels_file.writelines(yolo_labels_list)
        
                    yolo_labels_list.clear()
        
            labels_file.close()

if __name__ == "__main__":
    json_to_yolo = JSONToYOLO()

    json_to_yolo.transform_labels_to_yolo()
