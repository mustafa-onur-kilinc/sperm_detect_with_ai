"""
Resources used to write this script:
aturegano 2015, "Python Tkinter - resize widgets evenly in a window",
Stack Exchange Inc., accessed 13 May 2024,
<https://stackoverflow.com/a/30908374>

Özgün Zeki BOZKURT's detect.py and get_model.py scripts,
received 7 May 2024.

en-Knight 2016,  "In Python Tkinter, how do I set an OptionMenu to a 
fixed size shorter than the longest item?", Stack Exchange Inc., 
accessed 14 May 2024, <https://stackoverflow.com/a/35806211>

Combobox n.d., Python Software Foundation, accessed 14 May 2024,
<https://docs.python.org/3/library/tkinter.ttk.html#combobox>

Nihal Murmu 2023, "Displaying Image In Tkinter Python", C# Corner,
accessed 13 May 2024,
<https://www.c-sharpcorner.com/blogs/basics-for-displaying-image-in-tkinter-python>

Israel Dryer n.d., "Definitions", accessed 14 May 2024,
<https://ttkbootstrap.readthedocs.io/en/latest/themes/definitions/>

Glenn Jocher et. al. 2024, "Boxes", Ultralytics Inc., accessed 15 May 2024,
<https://docs.ultralytics.com/modes/predict/#boxes>

albert 2015, "List of All Tkinter Events", Stack Exchange Inc., 
accessed 15 May 2024, <https://stackoverflow.com/a/32289245>

John W. Shipman 2013, "54.3. Event types", New Mexico Tech, 
accessed 16 May 2024,
<https://web.archive.org/web/20190512164300id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-types.html>

John W. Shipman 2013, "54.5. Key names", New Mexico Tech, 
accessed 16 May 2024,
<https://web.archive.org/web/20190515021108id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/key-names.html>

John W. Shipman 2013, "54.6. Writing your handler: The Event class", 
New Mexico Tech, accessed 16 May 2024,
<https://web.archive.org/web/20190515013614id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-handlers.html>

John W. Shipman 2013, "54. Events: responding to stimuli", 
New Mexico Tech, accessed 16 May 2024,
<https://web.archive.org/web/20190509213522id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/events.html>

ohmu 2016, "How to clear text field part of ttk.Combobox?", 
Stack Exchange Inc., accessed 16 May 2024,
<https://stackoverflow.com/a/35236892>
"""

import os
import cv2
import yaml
import json
import re
import shutil
import torch
import tkinter
import tkinter.ttk

from tkinter import filedialog, messagebox
from tkinter import font 

from PIL import Image, ImageTk
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO


class NeuralNetworkGUI():
    """
    This class initializes a ttk window for GUI, lets user perform these
    actions by using GUI:
    - choose a folder of images to perform predictions on
    - perform predictions using YOLO, Faster R-CNN with ResNet50 
      backbone and RetinaNet with ResNet50 backbone
    - see label added image in a canvas
    - add, change or remove labels from image
    """

    def __init__(self):
        self.parent = tkinter.Tk()

        config_dirname = "config"
        yaml_name = "nn_gui_1.1_config.yaml"
        yaml_dir = os.path.join(config_dirname, yaml_name)

        with open(yaml_dir) as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        weights_dirname = args_dict["weights_dirname"]
        yolo_weight_dirname = args_dict["yolo_weight_dirname"]
        faster_rcnn_weight_dirname = args_dict["faster_rcnn_weight_dirname"]
        retina_net_weight_dirname = args_dict["retina_net_weight_dirname"]
        yolo_weight_name = args_dict["yolo_weight_name"]
        faster_rcnn_weight_name = args_dict["faster_rcnn_weight_name"]
        retina_net_weight_name = args_dict["retina_net_weight_name"]

        self.imgsz = args_dict["imgsz"]
        self.show = args_dict["show"]
        self.stream = args_dict["stream"]
        self.line_width = args_dict["line_width"]

        gui_background = args_dict["background"]
        gui_title = args_dict["title"]
        gui_geometry = args_dict["geometry"]
        self.model_options = args_dict["model_options"]
        self.save_options = args_dict["save_options"]
        self.label_options = args_dict["label_options"]

        # x_threshold and y_threshold are to prevent users from drawing
        # too small boxes
        self.x_threshold = args_dict["x_threshold"]
        self.y_threshold = args_dict["y_threshold"]
        self.text_distance_x = args_dict["text_distance_x"]
        self.text_distance_y = args_dict["text_distance_y"]

        self.primary_color = args_dict["primary_color"]
        self.success_color = args_dict["success_color"]
        self.warning_color = args_dict["warning_color"]
        self.danger_color = args_dict["danger_color"]
        self.active_color = args_dict["active_color"]
        self.highlight_bg_color = args_dict["highlight_bg_color"]

        self.parent.title(gui_title)
        self.parent.geometry(gui_geometry)
        self.parent.config(background=gui_background)

        self.label_id = tkinter.StringVar()

        self.init_window()

        self.cv_image = None
        self.image_on_canvas = None
        self.x0 = self.x1 = self.y0 = self.y1 = None
        self.chosen_label_x = self.chosen_label_y = None
        self.original_pred_labels_len = None

        self.pred_labels = []
        self.pred_labels_delete = []  # Holds a previous model's pred_labels
        self.pred_label_ids = []  # A list to easily check label_ids
        self.previous_frame_pred_labels = []
        self.images_list = []

        # Didn't use ternary operators to obey PEP8 maximum line length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  
        else:
            self.device = torch.device("cpu")

        self.transform = transforms.Compose(
            [
                transforms.Resize((1080, 1920)),
                # Convert images to PyTorch tensors with values in [0, 1]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
        
        self.script_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(self.script_dir, weights_dirname)
        self.yolo_weights_dir = os.path.join(weights_dir, yolo_weight_dirname,
                                             yolo_weight_name)
        self.faster_rcnn_weights_dir = os.path.join(weights_dir,
                                                    faster_rcnn_weight_dirname,
                                                    faster_rcnn_weight_name)
        self.retinanet_weights_dir = os.path.join(weights_dir, 
                                                  retina_net_weight_dirname,
                                                  retina_net_weight_name)

    def init_window(self):
        """
        Initializes self.parent's widgets, binds events to event handler 
        functions

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        self.parent.grid()
        arial25 = font.Font(family="Arial", size=25)

        labels_frame = tkinter.Frame(self.parent, background="gray25")
        labels_frame.pack(side="top")

        options_frame = tkinter.Frame(self.parent, background="gray25")
        options_frame.pack(side="top")

        left_arrow_frame = tkinter.Frame(self.parent, padx=10,
                                         background="gray25")
        left_arrow_frame.pack(side="left")
        
        right_arrow_frame = tkinter.Frame(self.parent, padx=10, 
                                          background="gray25")
        right_arrow_frame.pack(side="right")

        label_update_frame = tkinter.Frame(self.parent, padx=10,
                                           background="gray25")
        label_update_frame.pack(side="bottom")

        # label_row_col naming convention
        self.chosen_img_label = tkinter.Label(labels_frame, 
                                              text="Chosen Image:", 
                                              anchor="center",
                                              background="gray25", 
                                              foreground=self.primary_color)
        self.chosen_img_label.pack(fill="none", side="top", expand=True)

        self.pred_complete_label = tkinter.Label(labels_frame, text="", 
                                                 anchor="center", 
                                                 background="gray25", 
                                                 foreground=self.primary_color)
        self.pred_complete_label.pack(fill="none", side="top", expand=True)

        choose_img_folder_button = tkinter.Button(options_frame, 
                                                  text="Choose Folder", 
                                                  anchor="center", 
                                                  foreground="white",
                                                  background=self.primary_color,
                                                  activebackground=self.active_color, 
                                                  borderwidth=0, 
                                                  command=self.open_folder)
        choose_img_folder_button.pack(fill="none", padx=10, pady=10, ipadx=10, 
                               ipady=10, side="left", expand=True)

        pred_img_button = tkinter.Button(options_frame, text="Predict Image",
                                         anchor="center", foreground="white",
                                         background=self.primary_color, 
                                         activebackground=self.active_color,
                                         borderwidth=0,
                                         command=self.choose_predictor)
        pred_img_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10, 
                             side="left", expand=True)

        self.model_menu = tkinter.ttk.Combobox(options_frame, 
                                               values=self.model_options,
                                               background="gray90",
                                               state="readonly", width=12)
        self.model_menu.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10,
                             side="left", expand=True)
        self.model_menu.set(self.model_options[0])

        change_label_button = tkinter.Button(label_update_frame, 
                                             text="Change Label",
                                             anchor="center", foreground="white",
                                             background=self.warning_color,  
                                             activebackground=self.active_color,
                                             borderwidth=0,
                                             command=self.update_labels)
        change_label_button.pack(fill="none", padx=0, pady=10, ipadx=10, 
                                 ipady=10, side="left", expand=True)
        
        self.label_change_menu = tkinter.ttk.Combobox(label_update_frame, 
                                                      values=self.label_options,
                                                      background="gray90", 
                                                      state="readonly", 
                                                      width=12)
        self.label_change_menu.pack(fill="none", padx=20, pady=10, ipadx=10, 
                                    ipady=10, side="left", expand=True)
        self.label_change_menu.set(self.label_options[0])

        self.selected_box_label = tkinter.Label(label_update_frame, 
                                                  anchor="center",
                                                  text="Selected Box: ", 
                                                  background="gray25",
                                                  foreground=self.primary_color)
        self.selected_box_label.pack(fill="none", padx=20, pady=10, ipadx=10, 
                                  ipady=10, side="left", expand=True)
        
        label_id_entry = tkinter.Entry(label_update_frame, bg="gray90", 
                                       textvariable=self.label_id)
        label_id_entry.pack(fill="none", padx=20, pady=10, ipadx=10, 
                            ipady=10, side="left", expand=True)

        change_label_id_button = tkinter.Button(label_update_frame, text="Change ID",
                                          anchor="center", foreground="white",
                                          background=self.warning_color,  
                                          activebackground=self.active_color,
                                          borderwidth=0, 
                                          command=self.update_labels)
        change_label_id_button.pack(fill="none", padx=0, pady=10, ipadx=10, 
                                    ipady=10, side="left", expand=True)

        save_labels_button = tkinter.Button(options_frame, text="Save Labels", 
                                           anchor="center", foreground="white",
                                           background=self.success_color,
                                           activebackground=self.active_color, 
                                           borderwidth=0,
                                           command=self.save_preds_to_disk)
        save_labels_button.pack(fill="none", padx=10, pady=10, ipadx=10, 
                                ipady=10, side="left", expand=True)

        close_button = tkinter.Button(options_frame, text="Close",
                                      anchor="center", foreground="white",
                                      background=self.danger_color,
                                      activebackground=self.active_color,
                                      borderwidth=0, command=self.close_window)
        close_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10, 
                          side="left", expand=True)

        self.canvas = tkinter.Canvas(self.parent, background="gray25", borderwidth=0,
                                 highlightbackground=self.highlight_bg_color, 
                                 highlightthickness=2)
        self.canvas.pack(fill="both", side="top", padx=10, pady=10, expand=True)

        previous_img_button = tkinter.Button(left_arrow_frame, text="\u2190",
                                             anchor="center",
                                             foreground="white",
                                             background="#4c9be8",
                                             activebackground="#526170",
                                             borderwidth=0, font=arial25,
                                             command=self.open_previous_image)
        previous_img_button.pack(fill="none", side="left", expand=True)
        
        next_img_button = tkinter.Button(right_arrow_frame, text="\u2192",
                                         anchor="center", 
                                         foreground="white",
                                         background="#4c9be8",
                                         activebackground="#526170",
                                         borderwidth=0, font=arial25,
                                         command=self.open_next_image)
        next_img_button.pack(fill="none", side="right", expand=True)
        
        self.parent.bind("<KeyPress-Right>", self.get_arrow_keys)
        self.parent.bind("<KeyPress-Left>", self.get_arrow_keys)
        
        self.canvas.bind("<Configure>", self.resize_image)
        self.canvas.bind("<B1-Motion>", self.get_dragging_coords)
        self.canvas.bind("<ButtonRelease-1>", self.get_mouse_release_coords)
        self.canvas.bind("<Button-1>", self.get_mouse_click_coords)
        self.canvas.bind("<Button-3>", self.delete_labels)

    def open_folder(self):
        """
        Asks user to choose a file with Open function of OS, sends
        chosen image to self.cv_image

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        def extract_frame_number(filename):
            match = re.search(r'_(\d+)\.jpg$', filename)
            if match:
                return int(match.group(1))
            return float('inf')

        file_dir = ""

        self.folder_name = filedialog.askdirectory(initialdir=self.script_dir)
        
        if self.folder_name != "":
            self.images_list = os.listdir(self.folder_name)
            
            # Sort the list based on frame number
            self.images_list.sort(key=extract_frame_number)

            self.chosen_image_name = self.images_list[0]
            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            # Create labels directory
            self.labels_folder = self.folder_name + "_labels"

            if not os.path.exists(self.labels_folder):
                os.makedirs(self.labels_folder)
            else:
                # Made a string variable to obey 
                # PEP8 maximum line length
                msg = "The labels directory already exists. "
                msg += "Do you want to use the existing directory or overwrite it?"
                msg += "\n\nYes: Use existing\nNo: Overwrite\n"
                msg += "Cancel: Cancel operation"

                choice = messagebox.askyesnocancel(
                    title="Directory Exists",
                    message=msg
                )

                if choice is None:
                    return
                elif choice is False:
                    # Overwrite the directory
                    for file in os.listdir(self.labels_folder):
                        file_path = os.path.join(self.labels_folder, file)
                        try:
                            if (os.path.isfile(file_path) or 
                                    os.path.islink(file_path)):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            # Made a string variable to obey 
                            # PEP8 maximum line length
                            error_msg = f"Failed to delete {file_path}. "
                            error_msg += f"Reason: {e}"

                            messagebox.showerror(message=error_msg)
                    print("Overwriting directory:", self.labels_folder)
                else:
                    print("Using existing directory:", self.labels_folder)

            # Create subdirectories
            self.yolo_json_folder = os.path.join(self.labels_folder, 
                                                 'yolo_json')
            self.yolo_txt_folder = os.path.join(self.labels_folder, 
                                                'yolo_txt')
            self.cornercoordinates_json_folder = os.path.join(self.labels_folder, 
                                                              'cornercoordinates_json')
            self.cornercoordinates_txt_folder = os.path.join(self.labels_folder, 
                                                             'cornercoordinates_txt')

            os.makedirs(self.yolo_json_folder, exist_ok=True)
            os.makedirs(self.yolo_txt_folder, exist_ok=True)
            os.makedirs(self.cornercoordinates_json_folder, exist_ok=True)
            os.makedirs(self.cornercoordinates_txt_folder, exist_ok=True)

        else:
            self.labels_folder = ""

        # Resetting self.cv_image if an image has been opened before,
        # to prevent drawing labels on top of label drawn image
        if self.cv_image is not None:
            self.cv_image = None

        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        
        if file_dir != "":
            self.chosen_img_label.config(text=f"Chosen Image: {file_dir}")

            self.cv_image = cv2.imread(file_dir)
            self.cv_image = cv2.resize(self.cv_image, canvas_size)
                
            # Written self.pil_image to prevent Garbage Collector from
            # deleting function scope image
            # Read "Displaying Image In Tkinter Python" article in C# 
            # Corner website for more info, link in "Resources Used" 
            # at the top
            self.pil_image = ImageTk.PhotoImage(Image.fromarray(self.cv_image))
                    
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", 
                                                            image=self.pil_image,
                                                            tag="canvas_image")

            # Load labels for the chosen image
            self.load_labels()
            self.draw_labels()
        else:
            self.chosen_img_label.config(text=f"Chosen Image: ")
            # self.canvas.r  # self.canvas.r ?

    def load_labels(self):
        """
        Loads labels for the currently chosen image and the previous 
        image from the saved label files.
        
        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        base_filename = os.path.splitext(self.chosen_image_name)[0]
        yolo_json_path = os.path.join(self.yolo_json_folder, 
                                      f"{base_filename}.json")

        self.pred_labels = []

        if os.path.exists(yolo_json_path):
            with open(yolo_json_path, "r") as file:
                labels_list = json.load(file)
                self.pred_labels.append(self.chosen_image_name)
                self.pred_labels.append([
                    [label["id"], label["class"], label["x_middle"], 
                      label["y_middle"], label["width"], label["height"]]
                    for label in labels_list
                ])
        else:
            self.pred_labels.append(self.chosen_image_name)
            self.pred_labels.append([])

        # Load labels for the previous frame
        current_index = self.images_list.index(self.chosen_image_name)
        if current_index > 0:
            previous_image_name = self.images_list[current_index - 1]
            prev_base_filename = os.path.splitext(previous_image_name)[0]
            prev_yolo_json_path = os.path.join(self.yolo_json_folder, 
                                               f"{prev_base_filename}.json")

            self.previous_frame_pred_labels = []

            if os.path.exists(prev_yolo_json_path):
                with open(prev_yolo_json_path, "r") as file:
                    prev_labels_list = json.load(file)
                    self.previous_frame_pred_labels.append(previous_image_name)
                    self.previous_frame_pred_labels.append([
                        [label["id"], label["class"], label["x_middle"], 
                          label["y_middle"], label["width"], label["height"]]
                        for label in prev_labels_list
                    ])
            else:
                self.previous_frame_pred_labels.append(previous_image_name)
                self.previous_frame_pred_labels.append([])
        else:
            self.previous_frame_pred_labels = []
        
        print("Current frame labels: ", self.pred_labels)
        print("Previous frame labels: ", self.previous_frame_pred_labels)
        # print("====================================================")
        print("=" * 52)
            
    def get_arrow_keys(self, event):
        """
        Reads events received from widget it's binded to (self.parent),
        if event is user pressing right arrow, calls 
        self.open_next_image() function; if event is user pressing left
        arrow, calls self.open_previous_image() function. Does nothing
        for other user inputs

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """

        if event.keysym == "Right":
            self.open_next_image()
        elif event.keysym == "Left":
            self.open_previous_image()
    
    def open_previous_image(self):
        """
        Goes to previous image in image folder chosen by user, displays
        previous image on canvas

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.images_list == []:
            messagebox.showerror(title="No Image Folder Chosen",
                                message="No folder chosen to open images from.")
            return

        try:
            file_index = self.images_list.index(self.chosen_image_name)
        except ValueError as val_error:
            messagebox.showerror(title="Error While Accessing Image",
                                message=f"{val_error}")

        if file_index > 0:
            self.save_preds_to_disk()  # Automatically save labels

            # Prediction can't be complete after changing the image
            self.pred_complete_label.config(text="")

            i = 1
            self.chosen_image_name = self.images_list[file_index - i]
            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            # Finds first valid image in case there are missing images 
            # in folder
            while i <= file_index and not os.path.exists(file_dir):
                i += 1
                self.chosen_image_name = self.images_list[file_index - i]
                file_dir = os.path.join(self.folder_name,
                                        self.chosen_image_name)

            if os.path.exists(self.images_list[file_index - i]):
                self.chosen_image_name = self.images_list[file_index - i]

            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            # Resetting self.cv_image if an image has been opened before,
            # to prevent drawing labels on top of label drawn image
            if self.cv_image is not None:
                self.cv_image = None

            canvas_size = (self.canvas.winfo_width(), 
                           self.canvas.winfo_height())

            if file_dir != "":
                self.chosen_img_label.config(text=f"Chosen Image: {file_dir}")
                self.chosen_image_name = os.path.split(file_dir)[1]

                self.cv_image = cv2.imread(file_dir)
                self.cv_image = cv2.resize(self.cv_image, canvas_size)

                # Written self.pil_image to prevent Garbage Collector from
                # deleting function scope image
                # Read "Displaying Image In Tkinter Python" article in C# 
                # Corner website for more info, link in "Resources Used" 
                # at the top
                self.pil_image = ImageTk.PhotoImage(Image.fromarray(self.cv_image))

                self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw",
                                                                image=self.pil_image,
                                                                tag="canvas_image")

                # Load labels for the chosen image
                self.load_labels()
                self.draw_labels()
                
            else:
                self.chosen_img_label.config(text=f"Chosen Image: ")
                self.canvas.r
                    
    def open_next_image(self):
        """
        Goes to next image in image folder chosen by user, displays next
        image on canvas

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.images_list == []:
            messagebox.showerror(title="No Image Folder Chosen",
                                message="No folder chosen to open images from.")
            return

        try:
            file_index = self.images_list.index(self.chosen_image_name)
        except ValueError as val_error:
            messagebox.showerror(title="Error While Accessing Image",
                                message=f"{val_error}")

        if file_index < len(self.images_list) - 1:
            self.save_preds_to_disk()  # Automatically save labels

            self.pred_complete_label.config(text="")

            i = 1
            self.chosen_image_name = self.images_list[file_index + i]
            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            while (i < len(self.images_list) - file_index and 
                    not os.path.exists(file_dir)):
                print(f"i = {i}")
                i += 1
                self.chosen_image_name = self.images_list[file_index + i]
                file_dir = os.path.join(self.folder_name,
                                        self.chosen_image_name)

            if os.path.exists(self.images_list[file_index + i]):
                self.chosen_image_name = self.images_list[file_index + i]

            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            # Resetting self.cv_image if an image has been opened before,
            # to prevent drawing labels on top of label drawn image
            if self.cv_image is not None:
                self.cv_image = None

            canvas_size = (self.canvas.winfo_width(), 
                           self.canvas.winfo_height())

            if file_dir != "":
                self.chosen_img_label.config(text=f"Chosen Image: {file_dir}")
                self.chosen_image_name = os.path.split(file_dir)[1]

                self.cv_image = cv2.imread(file_dir)
                self.cv_image = cv2.resize(self.cv_image, canvas_size)

                # Written self.pil_image to prevent Garbage Collector from
                # deleting function scope image
                # Read "Displaying Image In Tkinter Python" article in C# 
                # Corner website for more info, link in "Resources Used" 
                # at the top
                self.pil_image = ImageTk.PhotoImage(Image.fromarray(self.cv_image))

                self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw",
                                                                image=self.pil_image,
                                                                tag="canvas_image")

                # Load labels for the chosen image
                self.load_labels()
                self.draw_labels()
            else:
                self.chosen_img_label.config(text=f"Chosen Image: ")
                self.canvas.r
            
    def resize_image(self, event):
        if self.cv_image is not None:
            new_width = event.width
            new_height = event.height
    
            resized_cv_image = cv2.resize(self.cv_image, (new_width, new_height))
            resized_pil_image = Image.fromarray(resized_cv_image)
            self.resized_photo_image = ImageTk.PhotoImage(resized_pil_image)
    
            self.canvas.itemconfig(self.image_on_canvas, 
                                   image=self.resized_photo_image)
            if self.pred_labels != []:

                for label in self.pred_labels[1]:
                    self.canvas.delete(f"rectangle_{label[0]}")
                    self.canvas.delete(f"text_{label[0]}")
        
                self.draw_labels()

    def get_dragging_coords(self, event):
        """
        Reads events received from widget it's binded to (self.canvas),
        if event is user clicking to widget with their left mouse button
        and dragging their mouse across widget, this function gets 
        starting coordinates of that event and updates the canvas to 
        dynamically show  the rectangle being drawn. Does nothing for 
        other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """

        # This if prevents getting new coordinates while dragging
        # mouse and only gives coordinates at the beginning
        if self.x0 is None and self.y0 is None:
            self.x0 = event.x
            self.y0 = event.y

        self.x1 = event.x
        self.y1 = event.y

        self.canvas.delete("current_rectangle")
        self.canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, 
                                     outline="red", tags="current_rectangle")

    def get_mouse_release_coords(self, event):
        """
        Reads events received from widget it's binded to (self.canvas),
        if event is user releasing their left mouse button after 
        dragging their mouse across widget, this function gets 
        coordinates of the point where user has released the mouse. 
        Calls self.draw_labels() function. If user was dragging the 
        mouse before, calls self.save_labels() function. Does nothing 
        for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """

        self.x1 = event.x
        self.y1 = event.y

        if self.x0 is not None and self.y0 is not None:
            self.save_labels()
        
        self.canvas.delete("current_rectangle")
        self.draw_labels()

    def get_mouse_click_coords(self, event):
        """
        Reads events received from widget it's binded to (self.canvas),
        if event is user clicking with their left mouse button, this 
        function gets coordinates of the point where user has clicked. 
        Calls self.update_labels() function. Does nothing for other user
        inputs

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """

        self.chosen_label_x = event.x
        self.chosen_label_y = event.y

        self.update_labels()

    def save_labels(self):
        """
        Gets mouse coordinates from self.get_dragging_coords() and 
        self.get_mouse_release_coords() functions, calculates values of 
        resulting bounding box in YOLO format, appends new label to 
        self.pred_labels list.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        if self.cv_image is not None:
            width = abs(self.x0 - self.x1)
            height = abs(self.y0 - self.y1)
            smaller_x = self.x0 if self.x0 < self.x1 else self.x1
            smaller_y = self.y0 if self.y0 < self.y1 else self.y1

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            x_middle = smaller_x + width // 2
            y_middle = smaller_y + height // 2
            normalized_width = width / canvas_width
            normalized_height = height / canvas_height
            normalized_x_middle = x_middle / canvas_width
            normalized_y_middle = y_middle / canvas_height

            pred_class = 0

            # Find the highest existing prediction ID
            if self.pred_labels and self.pred_labels[1]:
                highest_id = max(label[0] for label in self.pred_labels[1])
            else:
                highest_id = 0

            counter = highest_id + 1

            if (self.pred_labels != [] and self.pred_labels[1] == [] and 
                    width > self.x_threshold and height > self.y_threshold):
                # Enters this if block if user adds a single label and
                # decides to delete that labels afterwards
                
                self.pred_labels[1].append([counter, pred_class, 
                                        normalized_x_middle, 
                                        normalized_y_middle, 
                                        normalized_width, normalized_height])
                self.pred_label_ids.append(counter)
        
            elif (self.pred_labels != [] and self.pred_labels[1] != [] and
                    width > self.x_threshold and height > self.y_threshold):
                # Gets the highest label's counter and adds 1 to it
            
                self.pred_labels[1].append([counter, pred_class, 
                                            normalized_x_middle, 
                                            normalized_y_middle, 
                                            normalized_width, normalized_height])
                self.pred_label_ids.append(counter)
            
            # Set the selected box label to the new counter
            self.selected_box_label.config(text=f"Selected Box: {counter}")

            # Choose the newly created label to update without having to
            # click on it
            self.chosen_label_x = x_middle
            self.chosen_label_y = y_middle

            self.x0 = self.y0 = self.x1 = self.y1 = None

    def choose_predictor(self):
        """
        Chooses a prediction method according to user's choice of the
        option menu

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.cv_image is None:
            messagebox.showerror(message="No image chosen to predict.",
                                 title="No Image to Predict")
            return
        
        chosen_model = self.model_menu.get()
        self.pred_labels = []

        if chosen_model == "yolov8s":
            self.predict_with_yolo()
        elif chosen_model == "faster_rcnn":
            self.predict_with_faster_rcnn()
        elif chosen_model == "retina_net":
            self.predict_with_retina_net()
        else:
            messagebox.showerror(message="Unknown predictor!!!", 
                                 title="Unknown Choice")
            
    def predict_with_yolo(self):
        """
        Loads weight in self.yolo_weights_dir to YOLO, performs 
        prediction on self.cv_image 

        Parameters
        ----------
        Nones

        Returns
        ----------
        None
        """
        
        model = YOLO(self.yolo_weights_dir)

        results = model.predict(self.cv_image, stream=self.stream, 
                                imgsz=self.imgsz, show=self.show, 
                                line_width=self.line_width)
        
        self.pred_complete_label.config(text="Prediction complete.")

        new_predictions = []

        for result in results:
            boxes_cls = result.boxes.cls.cpu().numpy()
            boxes_xywhn = result.boxes.xywhn.cpu().numpy()

            for (box_cls, box_xywhn) in zip(boxes_cls, boxes_xywhn):
                new_predictions.append([int(box_cls), float(box_xywhn[0]), 
                                        float(box_xywhn[1]), float(box_xywhn[2]), 
                                        float(box_xywhn[3])])

        self.assign_ids(new_predictions)

    def predict_with_faster_rcnn(self, num_classes=3):
        """
        Loads weight in self.faster_rcnn_weights_dir to 
        fasterrcnn_resnet50_fpn_v2, performs prediction on PIL image
        derived from self.cv_image, saves predicted boxes and classes to
        self.pred_labels in YOLO format (to make boxes unaffected by
        changes in image size) 

        Parameters
        ----------
        num_classes : int, default=3
            Number of output classes (including background) for 
            FastRCNNPredictor

        Returns
        ----------
        None
        """

        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 
                                                        num_classes)
        model.load_state_dict(torch.load(self.faster_rcnn_weights_dir))

        model.eval()
        model.to(self.device)

        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        img_to_pred = Image.fromarray(rgb_image)

        img_tensor = self.transform(img_to_pred)
        with torch.no_grad():
            results = model([img_tensor.to(self.device)])

        self.pred_complete_label.config(text="Prediction complete.")

        new_predictions = []
        image_width = self.imgsz[0]
        image_height = self.imgsz[1]

        for result in results:
            boxes = result["boxes"].tolist()
            scores = result["scores"]
            classes = result["labels"].tolist()

            for (cls, score, box) in zip(classes, scores, boxes):
                if score > 0.5:
                    pred_class = 0 if cls == 1 else 1
                    
                    # Assigns values to x0, y0, x1, y1 in two lines
                    # to obey PEP8 maximum line length
                    x0, y0 = int(box[0]), int(box[1])
                    x1, y1 = int(box[2]), int(box[3])

                    width = abs(x0 - x1)
                    height = abs(y0 - y1)
                    x_middle = (x0 + x1) // 2
                    y_middle = (y0 + y1) // 2
                    normalized_x_middle = x_middle / image_width
                    normalized_y_middle = y_middle / image_height
                    normalized_width = width / image_width
                    normalized_height = height / image_height

                    new_predictions.append([pred_class, normalized_x_middle, 
                                            normalized_y_middle, 
                                            normalized_width, normalized_height])

        self.assign_ids(new_predictions)      

    def predict_with_retina_net(self, num_classes=3):
        """
        Loads weight in self.retinanet_weights_dir to 
        retinanet_resnet50_fpn_v2, performs prediction on PIL image
        derived from self.cv_image, saves predicted boxes and classes to
        self.pred_labels in YOLO format (to make boxes unaffected by
        changes in image size) 

        Parameters
        ----------
        num_classes : int, default=3
            Number of output classes (including background) for 
            RetinaNetClassificationHead

        Returns
        ----------
        None
        """

        model = retinanet_resnet50_fpn_v2(weights=None)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        model.load_state_dict(torch.load(self.retinanet_weights_dir))

        model.eval()
        model.to(self.device)

        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        img_to_pred = Image.fromarray(rgb_image)

        img_tensor = self.transform(img_to_pred)
        with torch.no_grad():
            results = model([img_tensor.to(self.device)])

        self.pred_complete_label.config(text="Prediction complete.")

        new_predictions = []
        image_width = self.imgsz[0]
        image_height = self.imgsz[1]

        for result in results:
            boxes = result["boxes"].tolist()
            scores = result["scores"]
            classes = result["labels"].tolist()

            for (cls, score, box) in zip(classes, scores, boxes):
                if score > 0.5:
                    pred_class = 0 if cls == 1 else 1
                    
                    # Assigns values to x0, y0, x1, y1 in two lines
                    # to obey PEP8 maximum line length
                    x0, y0 = int(box[0]), int(box[1])
                    x1, y1 = int(box[2]), int(box[3])

                    width = abs(x0 - x1)
                    height = abs(y0 - y1)
                    x_middle = (x0 + x1) // 2
                    y_middle = (y0 + y1) // 2
                    normalized_x_middle = x_middle / image_width
                    normalized_y_middle = y_middle / image_height
                    normalized_width = width / image_width
                    normalized_height = height / image_height

                    new_predictions.append([pred_class, normalized_x_middle, 
                                            normalized_y_middle, 
                                            normalized_width, normalized_height])

        self.assign_ids(new_predictions)

    def draw_labels(self):
        """
        Displays self.cv_image to user, draws labels to self.cv_image if
        self.pred_labels isn't empty list

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        if self.cv_image is None:
            messagebox.showerror(message="No image chosen to draw labels.",
                                  title="No Image to Draw")
            return
        
        # Prevents predictions from different models overlapping
        for label in self.pred_labels_delete:
            self.canvas.delete(f"rectangle_{label[0]}")
            self.canvas.delete(f"text_{label[0]}")

        self.parent.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if self.pred_labels != [] and \
              self.chosen_image_name == self.pred_labels[0]:
            for pred_label in self.pred_labels[1]:
                counter = pred_label[0]
                pred_class = pred_label[1]
                width = int(pred_label[4] * canvas_width)
                height = int(pred_label[5] * canvas_height)

                x0 = int(pred_label[2] * canvas_width) - width // 2
                x1 = int(pred_label[2] * canvas_width) + width // 2
                y0 = int(pred_label[3] * canvas_height) - height // 2
                y1 = int(pred_label[3] * canvas_height) + height // 2

                text_location_x = x0 - self.text_distance_x
                text_location_y = y0 - self.text_distance_y

                rect_color = "#66BB6A" if pred_class == 0 else "blue"
    
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             tags=f"rectangle_{counter}",
                                             outline=rect_color,
                                             width = 3,
                                             activeoutline=self.warning_color)
                self.canvas.create_text([text_location_x, text_location_y],
                                        text=f"{counter}", 
                                        tags=f"text_{counter}",
                                        fill='red', font=("Arial", 12))
                
            self.pred_labels_delete = self.pred_labels[1].copy()

    def update_labels(self):
        """
        Gets left click coordinates from self.get_mouse_click_coords()
        function, checks if clicked location is within a bounding box, 
        if user has clicked a bounding box, lets user change class and 
        id of that bounding box's label. Calls self.draw_labels() to 
        display new labels to user

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        if self.cv_image is None:
            messagebox.showerror(title="No Image Selected",
                                 message="No image selected to update labels of.")
            return
        
        if self.chosen_label_x is None or self.chosen_label_y is None:
            messagebox.showerror(title="No Label Selected", 
                                 message="No label selected to update.")
            return
        
        click_x = self.chosen_label_x
        click_y = self.chosen_label_y
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if self.pred_labels != []:
            for label in self.pred_labels[1]:
                width = int(label[4] * canvas_width)
                height = int(label[5] * canvas_height)
                x0 = int(label[2] * canvas_width) - width // 2            
                x1 = int(label[2] * canvas_width) + width // 2
                y0 = int(label[3] * canvas_height) - height // 2
                y1 = int(label[3] * canvas_height) + height // 2

                is_in_x = (click_x - x0) < width and (x1 - click_x) < width
                is_in_y = (click_y - y0) < height and (y1 - click_y) < height

                if is_in_x and is_in_y:
                    self.selected_box_label.config(text=f"Selected Box: {label[0]}")
                    
                    if self.label_change_menu.get() == "Sperm":
                        label[1] = 0

                        self.label_id.set("")
                        self.label_change_menu.set(self.label_options[0])
                    elif self.label_change_menu.get() == "Non-Sperm":
                        label[1] = 1

                        self.label_id.set("")
                        self.label_change_menu.set(self.label_options[0])
                    
                    if self.label_id.get() != "":
                        i = 0
                        label_id = int(self.label_id.get())

                        self.selected_box_label.config(text=f"Selected Box: {label[0]}")

                        while (label_id != self.pred_labels[1][i][0] and 
                               i < len(self.pred_labels[1]) - 1):
                            i += 1
                        
                        if (i >= len(self.pred_labels[1]) - 1 and
                                label_id != self.pred_labels[1][-1][0]):
                            self.canvas.delete(f"text_{label[0]}")
                            label[0] = label_id
                            
                        self.selected_box_label.config(text=f"Selected Box: {label[0]}")

                        self.label_id.set("")
                        self.label_change_menu.set(self.label_options[0])

                    break

        self.draw_labels()

    def delete_labels(self, event):
        """
        Reads events received from widget it's binded to (self.canvas),
        if event is user clicking with their right mouse button, this 
        function gets coordinates of the point where user has clicked. 
        If right clicked location is within a bounding box, deletes that 
        bounding box and its id from self.pred_labels list and canvas. 
        Does nothing for other user inputs

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """
        
        if self.cv_image is None:
            return
        
        click_x = event.x
        click_y = event.y
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if self.pred_labels != []:
            for label in self.pred_labels[1]:
                width = int(label[4] * canvas_width)
                height = int(label[5] * canvas_height)
                x0 = int(label[2] * canvas_width) - width // 2            
                x1 = int(label[2] * canvas_width) + width // 2
                y0 = int(label[3] * canvas_height) - height // 2
                y1 = int(label[3] * canvas_height) + height // 2

                is_in_x = (click_x - x0) < width and (x1 - click_x) < width
                is_in_y = (click_y - y0) < height and (y1 - click_y) < height

                if is_in_x and is_in_y:
                    self.pred_labels[1].remove(label)
                    self.canvas.delete(f"rectangle_{label[0]}")
                    self.canvas.delete(f"text_{label[0]}")
                    break

    def save_preds_to_disk(self):
        """
        Automatically saves labels in different formats and directories

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.cv_image is None:
            messagebox.showerror(message="No image chosen to save labels of.",
                                title="No Image to Save Labels")
            return
        '''
        if self.pred_labels == []:
            messagebox.showerror(message="No prediction performed to save labels of.",
                                title="No Label to Save")
            return
        '''

        base_filename = os.path.splitext(self.chosen_image_name)[0]

        # Save in YOLO txt format
        self.save_txt(os.path.join(self.yolo_txt_folder, 
                                   f"{base_filename}.txt"), is_yolo_format=True)
        # Save in corner coordinates txt format
        self.save_txt(os.path.join(self.cornercoordinates_txt_folder, 
                                   f"{base_filename}.txt"), is_yolo_format=False)
        # Save in YOLO json format
        self.save_json(os.path.join(self.yolo_json_folder, 
                                    f"{base_filename}.json"), 
                        is_yolo_format=True)
        # Save in corner coordinates json format
        self.save_json(os.path.join(self.cornercoordinates_json_folder, 
                                    f"{base_filename}.json"), 
                        is_yolo_format=False)
    
    def save_txt(self, path, is_yolo_format: bool = False):
        """
        Saves labels in TXT format

        If is_yolo_format is True, labels are saved in this format:
        class x_middle y_middle width height
        (x_middle, y_middle, width and height are normalized to image
        width and height)

        If is_yolo_format is False, labels are saved in this format:
        class x_start y_start x_end y_end

        Parameters
        ----------
        path : str
            Path to save the txt file
        is_yolo_format : bool, default=False
            Shows whether TXT file should be saved in YOLO format or not

        Returns
        ----------
        None
        """

        labels_str = ""

        if self.pred_labels != []:
            for pred_label in self.pred_labels[1]:
                pred_class = pred_label[1]
                x_middle = pred_label[2]
                y_middle = pred_label[3]
                box_width = pred_label[4]
                box_height = pred_label[5]

                if is_yolo_format:
                    labels_str += f"{pred_class} {x_middle} {y_middle}"
                    labels_str += f"{box_width} {box_height}\n"
                else:
                    image_width = self.imgsz[0]
                    image_height = self.imgsz[1]

                    x_middle_px = int(x_middle * image_width)
                    y_middle_px = int(y_middle * image_height)

                    box_width_px = int(box_width * image_width)
                    box_height_px = int(box_height * image_height)

                    x_start = x_middle_px - box_width_px // 2
                    y_start = y_middle_px - box_height_px // 2
                    x_end = x_middle_px + box_width_px // 2
                    y_end = y_middle_px + box_height_px // 2

                    labels_str += f"{pred_class} {x_start} {y_start}"
                    labels_str += f"{x_end} {y_end}\n"

        if path:
            try:
                with open(path, "w+") as file:
                    file.write(labels_str)
            except Exception as e:
                messagebox.showerror(message=f"Error: {e}",
                                     title="Error While Saving TXT")
    
    def save_json(self, path, is_yolo_format: bool = False):
        """
        Saves labels in JSON format

        Parameters
        ----------
        path : str
            Path to save the json file
        is_yolo_format : bool, default=False
            Shows whether JSON file should be saved in YOLO format or not

        Returns
        ----------
        None
        """
        
        labels_list = []
        if self.pred_labels != []:
            for pred_label in self.pred_labels[1]:
                labels_dict = {}

                counter = pred_label[0]
                pred_class = pred_label[1]
                x_middle = pred_label[2]
                y_middle = pred_label[3]
                box_width = pred_label[4]
                box_height = pred_label[5]

                if is_yolo_format:
                    labels_dict = {
                        "id": counter,
                        "class": pred_class,
                        "x_middle": x_middle,
                        "y_middle": y_middle,
                        "width": box_width,
                        "height": box_height
                    }
                else:
                    image_width = self.imgsz[0]
                    image_height = self.imgsz[1]

                    x_middle_px = int(x_middle * image_width)
                    y_middle_px = int(y_middle * image_height)

                    box_width_px = int(box_width * image_width)
                    box_height_px = int(box_height * image_height)

                    x_start = x_middle_px - box_width_px // 2
                    y_start = y_middle_px - box_height_px // 2
                    x_end = x_middle_px + box_width_px // 2
                    y_end = y_middle_px + box_height_px // 2

                    labels_dict = {
                        "id": counter,
                        "class": pred_class,
                        "x_start": x_start,
                        "y_start": y_start,
                        "x_end": x_end,
                        "y_end": y_end
                    }

                labels_list.append(labels_dict)

        if path:
            try:
                with open(path, "w+") as file:
                    json.dump(labels_list, file, indent=4)
            except Exception as e:
                messagebox.showerror(message=f"Error: {e}",
                                    title="Error While Saving JSON")

    def close_window(self):
        """
        Asks user an "Are you sure you want to exit?" question, if answer
        is Yes, closes app. If answer is No or no answer is chosen, does 
        nothing

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        choice = messagebox.askyesno(title="Exiting...",
                                     message="Are you sure you want to exit?")
        
        if choice:
            self.parent.destroy()
    
    
    def assign_ids(self, new_predictions: list):
        """
        Assigns IDs to labels in new_predictions by using cost matrix
        to best match new_predictions with 
        self.previous_frame_pred_labels and establish ID continuity

        Parameters
        ----------
        new_predictions : list
            Includes new_predictions to assign values

        Returns
        ----------
        None
        """
        
        if not self.previous_frame_pred_labels:
            # First frame or no previous predictions
            counter = 1

            for pred in new_predictions:
                pred.insert(0, counter)
                counter += 1
                
            self.pred_labels = [self.chosen_image_name, new_predictions]
        else:
            cost_matrix = self.calculate_cost_matrix(new_predictions, 
                                                     self.previous_frame_pred_labels)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_ids = set()

            # Broke the ternary operator used here to obey PEP8 maximum
            # line length (Sorry for that :( )
            if self.previous_frame_pred_labels:
                max_id = max([label[0] for label in self.previous_frame_pred_labels[1]])
            else:
                max_id = 0

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i][j] < 1.0:  # Threshold for assignment
                    new_predictions[i].insert(0, self.previous_frame_pred_labels[1][j][0])
                    assigned_ids.add(self.previous_frame_pred_labels[1][j][0])
                else:
                    max_id += 1
                    new_predictions[i].insert(0, max_id)

            # Assign new IDs to unmatched predictions
            for i in range(len(new_predictions)):
                if len(new_predictions[i]) == 5:
                    max_id += 1
                    new_predictions[i].insert(0, max_id)

            self.pred_labels = [self.chosen_image_name, new_predictions]

        self.previous_frame_pred_labels = self.pred_labels.copy()
        self.draw_labels()

    def calculate_cost_matrix(self, new_predictions, previous_predictions):
        """
        Calculates cost matrix by using IoU values of new predictions
        and previous predictions

        Parameters
        ----------
        new_predictions : list
            New predictions to calculate IoU with previous predictions
        previous_predictions : list
            Previous predictions to calculate IoU with new predictions

        Returns
        ----------
        cost_matrix : list
            Contains IoU values of every new prediction with its 
            corresponding previous prediction in matrix format
        """

        cost_matrix = []

        for new_pred in new_predictions:
            row = []

            for prev_pred in previous_predictions[1]:
                iou = self.calculate_iou(new_pred, prev_pred)
                row.append(1 - iou)  # Use 1 - IoU for cost matrix

            cost_matrix.append(row)

        return cost_matrix

    def calculate_iou(self, boxA, boxB):
        """
        Calculates IoU (Intersection over Union) for two bounding boxes
        given, returns IoU value

        Parameters
        ----------
        boxA : tuple
            Contains bounding box information for first bounding box in 
            [class, x_center, y_center, width, height] format
        boxB : tuple
            Contains bounding box information for second bounding box in 
            [class, x_center, y_center, width, height] format

        Returns
        ----------
        iou : Float
            Intersection over Union value of boxA and boxB
        """
        
        # boxA and boxB are in the format 
        # [class, x_center, y_center, width, height]
        xA1, yA1, wA, hA = boxA[1:5]
        xB1, yB1, wB, hB = boxB[2:6]

        xA1 -= wA / 2
        yA1 -= hA / 2
        xA2 = xA1 + wA
        yA2 = yA1 + hA

        xB1 -= wB / 2
        yB1 -= hB / 2
        xB2 = xB1 + wB
        yB2 = yB1 + hB

        interX1 = max(xA1, xB1)
        interY1 = max(yA1, yB1)
        interX2 = min(xA2, xB2)
        interY2 = min(yA2, yB2)

        interArea = max(0, interX2 - interX1) * max(0, interY2 - interY1)
        boxAArea = wA * hA
        boxBArea = wB * hB

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


if __name__ == "__main__":
    nn_gui = NeuralNetworkGUI()

    nn_gui.parent.mainloop()