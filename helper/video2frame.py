"""
This Python script is written to read a video file and save frames from 
that video as images.

In order to work, this script should be in the same folder as the folder 
of videos. For example:

- main_dir
    - helpers_folder
        - video2frame.py
    - config_folder
        - video2frame.yaml
    - dataset_folder
        - Patient_1
            - Ekran
                - video_name
                - (...)
            - Telefon
                - video_name
                - (...)
        - Patient_2
            - (...)

Resources used to write this script:
OpenCV n.d., "Getting Started with Videos", accessed 25 March 2024,
<https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html>

OpenCV n.d., "cv::VideoCapture Class Reference", accessed 25 March 2024,
<https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>

nikhilagg 2020, "Create a directory in Python", 
GeeksforGeeks, accessed 25 March 2024, 
<https://www.geeksforgeeks.org/create-a-directory-in-python/>

Simplilearn 2023, 
"Python Check if File Exists: How to Check if a Directory Exists?", 
Simplilearn Solutions, accessed 25 March 2024, 
<https://www.simplilearn.com/tutorials/python-tutorial/python-check-if-file-exists>

John Sturtz n.d., "Modulo String Formatting in Python", 
Real Python, accessed 25 March 2024, 
<https://realpython.com/python-modulo-string-formatting/>

Hieru Tran Trung 2017, 
"OpenCV - Saving images to a particular folder of choice", 
Stack Overflow Inc., accessed 25 March 2024, 
<https://stackoverflow.com/a/41626482> 
"""

import os
import cv2
import yaml

class VideoToFrame():
    def __init__(self) -> None:
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]

        config_dirname = "config"
        yaml_name = "video2frame.yaml"
        yaml_dir = os.path.join(main_dir, config_dirname, yaml_name)

        with open(yaml_dir, "r", encoding="utf8") as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        dataset_name = args_dict["dataset_name"]
        
        src_name = args_dict["src_name"]

        is_src_name_telefon = args_dict["is_src_name_telefon"]  
        
        # Using if-else block to make choosing Telefon/Ekran less prone
        # to error and easier
        if is_src_name_telefon:
            src_name.append("Telefon")
        else:
            src_name.append("Ekran")
            
        self.video_name = args_dict["video_name"]
        
        dataset_dir = os.path.join(main_dir, dataset_name)

        if not os.path.exists(dataset_dir):
            print(f"{dataset_dir} isn't an existing folder!!!")
            exit()
        
        # Enter folder names of patients and Ekran/Telefon here
        video_folder_dir = os.path.join(dataset_dir, src_name[0], src_name[1])
        
        if not os.path.exists(video_folder_dir):
            print(f"{video_folder_dir} isn't an existing folder!!!")
            exit()
        
        self.video_file_dir = os.path.join(video_folder_dir, self.video_name)
        
        if not os.path.exists(self.video_file_dir):
            print(f"{self.video_file_dir} isn't an existing file!!!")
            exit()
        
        # self.images_folder_dir is the directory to save frames
        self.images_folder_dir = os.path.join(video_folder_dir, 
                                              self.video_name.split(".")[0])
        
        try:
            if not os.path.exists(self.images_folder_dir):
                os.mkdir(self.images_folder_dir)
        except OSError as e:
            print(e)
            exit()

    def frames_from_video(self):
        """
        Reads video file in self.video_file_dir, saves each frame from
        video to self.images_folder_dir

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        capturer = cv2.VideoCapture(self.video_file_dir)
        
        if not capturer.isOpened():
            print("Couldn't open video file. Exiting...")
            exit()
        
        counter = 1
        os.chdir(self.images_folder_dir)
        
        while True:
            retVal, frame = capturer.read()
        
            if not retVal:
                print("Can't receive frame. The video stream may have ended.")
                print("Exiting...")
                break
        
            frame_name = self.video_name.split(".")[0]
            frame_name = frame_name + "_" + str(counter) + ".png"
        
            isWritten = cv2.imwrite(frame_name, frame)
        
            if isWritten:
                print(f"Written {frame_name} frame to {self.images_folder_dir}")
        
            counter += 1
        
        capturer.release()

if __name__ == "__main__":
    video_to_frame = VideoToFrame()

    video_to_frame.frames_from_video()