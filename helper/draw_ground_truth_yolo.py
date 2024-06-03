"""
This script adds ground truth labels in YOLO format to images and videos

This script requires a directory like this to work properly:
- main_dir
    - helpers_folder
        - draw_ground_truth_yolo.py
    - config_folder
        - draw_ground_truth_yolo.yaml
    - runs_dir_name
         - detect_dir_name
              - predict_dir_name
                   - pred_video_name
                   - pred_images_name
                        - video_name_1.png
                        - video_name_2.png
                        - video_name_3.png
                        - (...)
    - dataset_dir_name
         - src_dir
              - Ekran
                   - video_name
                        - video_name_1.png
                        - video_name_2.png
                        - video_name_3.png
                        - (...)
                   - labels_dir_name
                        - video_name_1.txt
                        - video_name_2.txt
                        - video_name_3.txt
                        - (...)
              - Telefon
                   - video_name
                        - video_name_1.png
                        - video_name_2.png
                        - video_name_3.png
                        - (...)
                   - labels_dir_name
                        - video_name_1.txt
                        - video_name_2.txt
                        - video_name_3.txt
                        - (...)
    
Resources used while writing this script:
Avinab Saha 2017, "Read, Write and Display a video using OpenCV", 
Big Vision LLC, accessed 26 April 2024, 
<https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/>

"fourcc: DIVX", fourcc.org, accessed 26 April 2024, 
<https://fourcc.org/divx/>

"Getting Started with Videos" n.d., accessed 26 April 2024,
<https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html>
"""

import os
import cv2
import yaml

class DrawGroundTruthYOLO():
    def __init__(self) -> None:
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]
    
        config_dirname = "config"
        yaml_name = "draw_ground_truth_yolo.yaml"
        yaml_dir = os.path.join(main_dir, config_dirname, yaml_name)
    
        with open(yaml_dir, "r", encoding="utf8") as yaml_file:
            args_dict = yaml.safe_load(yaml_file)
    
        runs_dir_name = args_dict["runs_dir_name"]
        detect_dir_name = args_dict["detect_dir_name"]
        predict_dir_name = args_dict["predict_dir_name"]
        dataset_dir_name = args_dict["dataset_dir_name"]
        src_dir = args_dict["src_dir"]
        is_src_dir_Telefon = args_dict["is_src_dir_Telefon"]
        self.video_name = args_dict["video_name"]
        self.save_ground_truth_images = args_dict["save_ground_truth_images"]
    
        labels_dir_name = self.video_name + "_" + "labels"
        pred_video_name = self.video_name + ".avi"
        pred_images_name = self.video_name + "_" + "images"
        gt_images_name = self.video_name + "_" + "gt" + "_" + "images"
        
        if is_src_dir_Telefon:
            src_dir.append("Telefon") 
        else:
            src_dir.append("Ekran")
    
        detect_dir = os.path.join(main_dir, runs_dir_name, detect_dir_name)
        self.predict_dir = os.path.join(detect_dir, predict_dir_name)
        pred_video_dir = os.path.join(self.predict_dir, pred_video_name)
        self.pred_images_dir = os.path.join(self.predict_dir, pred_images_name)
        self.gt_images_dir = os.path.join(self.predict_dir, gt_images_name)
    
        dataset_dir = os.path.join(main_dir, dataset_dir_name)
    
        patient_dir = os.path.join(dataset_dir, src_dir[0], src_dir[1])
        self.labels_dir = os.path.join(patient_dir, labels_dir_name)
        images_dir = os.path.join(patient_dir, self.video_name)
    
        if not os.path.exists(self.gt_images_dir):
            os.mkdir(self.gt_images_dir)

        self.labels_count = len(os.listdir(self.labels_dir))
    
    def convert_yolo_labels(self, labels_file, image_size: list = [640, 640]):
        """
        Converts YOLO labels from labels_file to 
        [class_id, x_left, y_top, width, height] format
    
        Parameters
        ----------
        labels_file: TXT file
            TXT file that includes YOLO labels
        image_size: list[int, int], default=[640, 640]
            Size of image in [width, height] format
    
        Returns
        ----------
        labels_list: list
            Converted labels of every label in labels_file
    
        Raises
        ----------
        OSError
            If labels_file can't be opened
        """
        
        image_width = image_size[0]
        image_height = image_size[1]
    
        labels_list = []
        
        with open(labels_file, "r") as file:
            for line in file.readlines():
                class_id, x_middle, y_middle, bb_width, bb_height = line.split()
                
                bb_width = int(float(bb_width) * int(image_width))
                bb_height = int(float(bb_height) * int(image_height))
    
                x_middle = int(float(x_middle) * int(image_width))
                y_middle = int(float(y_middle) * int(image_height))
    
                x_left = x_middle - (bb_width // 2)
                y_top = y_middle - (bb_height // 2)
    
                labels_list.append([class_id, x_left, y_top, bb_width, bb_height])
    
        return labels_list
    
    def draw_ground_truth_img(self, images_dir: str, image_name: str, 
                              save_dir: str, labels_list: list):
        """
        Draws ground truth labels from labels_list to image_name located in 
        images_dir and displays resulting image
    
        Parameters
        ----------
        images_dir: str
            Directory of images to draw ground truth labels on
        image_name: str
            Name of image to draw ground truth labels on
        save_dir: str
            Directory of images to save ground truth images to
        labels_list: list
            Ground truth labels in [class_id, x_left, y_top, width, height] 
            format
    
        Returns
        ----------
        None
        """
        
        os.chdir(images_dir)
    
        drawn_image = cv2.imread(image_name)
    
        for j in range(len(labels_list)):
            point_1 = (labels_list[j][1], labels_list[j][2])
            point_2 = (labels_list[j][1] + labels_list[j][3], 
                       labels_list[j][2] + labels_list[j][4])
            
            color = (0, 255, 0) if labels_list[j][0] == "0" else (0, 255, 255)
    
            drawn_image = cv2.rectangle(drawn_image, point_1, point_2, 
                                        color=color, thickness=2)
    
        os.chdir(save_dir)
        
        cv2.imwrite(image_name, drawn_image)
        print("Image successfully saved")
        
        cv2.imshow("Ground Truth Drawn Image", drawn_image)
        cv2.waitKey(1)
    
    def draw_ground_truth_vid(self, videos_dir: str, video_file_name: str, 
                              fps: int = 10, video_size: list = [640, 640]):
        """
        Draws ground truth labels to video_name located in videos_dir, 
        displays and saves resulting video
    
        Parameters
        ----------
        videos_dir: str
            Directory of video(s) to draw ground truth labels on
        video_file_name: str
            Name of video to draw ground truth labels on
        fps: int, default=10
            FPS value of resulting video to save
        video_size: list, default=[640, 640]
            Size of video in [width, height] format
    
        Returns
        ----------
        None
        """
        
        os.chdir(videos_dir)
    
        save_video_name = video_file_name.split(".")[0] + "_ground_truth" + \
                            "." + video_file_name.split(".")[1]
        
        # Windows 11 Media Player can't open MJPG format, 
        # can open DIVX format though
        fourcc = cv2.VideoWriter.fourcc('D', 'I', 'V', 'X')
    
        cap = cv2.VideoCapture(video_file_name)
        out = cv2.VideoWriter(save_video_name, fourcc=fourcc, fps=fps,
                              frameSize=(video_size[0], video_size[1]))
    
        if not cap.isOpened:
            print("Error opening video!!!")
            return
    
        i = 0
    
        while cap.isOpened:
            ret, frame = cap.read()
    
            if ret:
                print("Read video successfully")
    
                label_file_name = self.video_name + "_" + str(i + 1) + ".txt"
                label_dir = os.path.join(self.labels_dir, label_file_name)
    
                labels_list = self.convert_yolo_labels(label_dir, 
                                                       image_size=[1920, 1080])
    
                for j in range(len(labels_list)):
                    point_1 = (labels_list[j][1], labels_list[j][2])
                    point_2 = (labels_list[j][1] + labels_list[j][3], 
                               labels_list[j][2] + labels_list[j][4])
                    
                    if labels_list[j][0] == "0":
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
            
                    frame = cv2.rectangle(frame, point_1, point_2, color=color, 
                                          thickness=7)
        
                out.write(frame)
                print("Successfully written frame")
                
                cv2.imshow("Ground Truth Drawn Frame", frame)
                cv2.waitKey(1)
    
                i += 1
            else:
                # Assuming video has ended if ret == False
                print("Could not read video (has video ended?)")
                break
    
        cap.release()
        out.release()

    def main_function(self):
        if self.save_ground_truth_images:
            for i in range(self.labels_count):
                label_file_name = self.video_name + "_" + str(i + 1) + ".txt"
                image_file_name = self.video_name + "_" + str(i + 1) + ".png"
        
                label_dir = os.path.join(self.labels_dir, label_file_name)
                labels_list = self.convert_yolo_labels(label_dir, 
                                                       image_size=[1920, 1080])
                
                self.draw_ground_truth_img(images_dir=self.pred_images_dir, 
                                      image_name=image_file_name,
                                      save_dir=self.gt_images_dir, 
                                      labels_list=labels_list)
        else:  
            video_file_name = self.video_name + ".avi"
        
            self.draw_ground_truth_vid(videos_dir=self.predict_dir, 
                                  video_file_name=video_file_name,
                                  video_size=[1920, 1080])


if __name__ == "__main__":
    draw_gt_yolo = DrawGroundTruthYOLO()

    draw_gt_yolo.main_function()

    cv2.destroyAllWindows()    
