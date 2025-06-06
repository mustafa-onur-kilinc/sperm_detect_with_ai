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

Glenn Jocher et. al. 2024, "Boxes", Ultralytics Inc., accessed 25 May 2024,
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
New Mexico Tech, accessed 22 May 2024,
<https://web.archive.org/web/20190515013614id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/event-handlers.html>

John W. Shipman 2013, "54. Events: responding to stimuli", 
New Mexico Tech, accessed 16 May 2024,
<https://web.archive.org/web/20190509213522id_/http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/events.html>

ohmu 2016, "How to clear text field part of ttk.Combobox?", 
Stack Exchange Inc., accessed 16 May 2024,
<https://stackoverflow.com/a/35236892>

Bernd Klein 2022, "9. Sliders in Tkinter", accessed 22 May 2024,
<https://python-course.eu/tkinter/sliders-in-tkinter.php>

John W. Shipman 2013, "21. The Scale widget", accessed 23 May 2024,
<https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/scale.html>

unutbu 2013, "How to bind Ctrl+/ in python tkinter?", 
Stack Exchange, Inc., accessed 22 May 2024, 
<https://stackoverflow.com/a/16082411>

newtocoding 2021, "Python – Tkinter Choose color Dialog", GeeksforGeeks, 
accessed 23 May 2024, 
<https://www.geeksforgeeks.org/python-tkinter-choose-color-dialog/>

Dr. Jan-Philip Gehrcke 2012, "write() versus writelines() and 
concatenated strings", Stack Exchange, Inc., accessed 24 May 2024, 
<https://stackoverflow.com/a/12377575>

Curt Hagenloger 2009, "Best method for reading newline delimited files 
and discarding the newlines?", Stack Exchange, Inc., 
accessed 24 May 2024, <https://stackoverflow.com/a/544932>

Dev Prakash Sharma 2021, "How to clear Tkinter Canvas?", 
Tutorials Point, accessed 25 May 2024,
<https://www.tutorialspoint.com/how-to-clear-tkinter-canvas>

wm attributes 2018, accessed 31 May 2024, 
<https://wiki.tcl-lang.org/page/wm+attributes>
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import os
import cv2
import yaml
import json
import re
import shutil
import torch
import tkinter
import tkinter.ttk


import threading
from tkinter import filedialog, messagebox
from tkinter import font
from tkinter import colorchooser

from PIL import Image, ImageTk
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial
from scipy.optimize import linear_sum_assignment

from ultralytics import YOLO

from sperm_detection.tracker import Tracker

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
        yaml_name = "nn_gui_1.2_config.yaml"
        yaml_dir = os.path.join(config_dirname, yaml_name)

        with open(yaml_dir) as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        # weights_dirname = args_dict["weights_dirname"]
        # yolo_weight_dirname = args_dict["yolo_weight_dirname"]
        # faster_rcnn_weight_dirname = args_dict["faster_rcnn_weight_dirname"]
        # retina_net_weight_dirname = args_dict["retina_net_weight_dirname"]
        # yolo_weight_name = args_dict["yolo_weight_name"]
        # faster_rcnn_weight_name = args_dict["faster_rcnn_weight_name"]
        # retina_net_weight_name = args_dict["retina_net_weight_name"]

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
        self.label_colors = args_dict["label_colors"]

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
        self.frame_num = tkinter.IntVar()
        self.new_label_cls = tkinter.StringVar()

        self.init_window()

        self.sort_tracking_enabled = False
        self.cv_image = None
        self.image_on_canvas = None
        self.x0 = self.x1 = self.y0 = self.y1 = None
        self.chosen_label_x = self.chosen_label_y = None
        self.original_pred_labels_len = None
        self.chosen_model = ""
        
        self.selected_label = None
        self.pred_labels = []
        self.pred_labels_delete = []  # Holds a previous model's pred_labels
        self.pred_label_ids = []  # A list to easily check label_ids
        self.previous_frame_pred_labels = []
        self.copied_label = []  # To hold a label copied by user
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
        # self.weights_dir = os.path.join(self.script_dir, weights_dirname)
        # self.yolo_weights_dir = os.path.join(self.weights_dir, yolo_weight_dirname,
        #                                      yolo_weight_name)
        # self.faster_rcnn_weights_dir = os.path.join(self.weights_dir,
        #                                             faster_rcnn_weight_dirname,
        #                                             faster_rcnn_weight_name)
        # self.retinanet_weights_dir = os.path.join(self.weights_dir, 
        #                                           retina_net_weight_dirname,
        #                                           retina_net_weight_name)


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

        slider_frame = tkinter.Frame(self.parent, padx=10,
                                     background="gray25")
        slider_frame.pack(side="bottom")

        self.chosen_img_label = tkinter.Label(labels_frame, 
                                              text="Chosen Image:", 
                                              anchor="center",
                                              background="gray25", 
                                              foreground=self.primary_color)
        self.chosen_img_label.pack(fill="none", side="top", expand=True)

        self.chosen_model_label = tkinter.Label(labels_frame, 
                                                text="Chosen Model:", 
                                                anchor="center", 
                                                background="gray25", 
                                                foreground=self.primary_color)
        self.chosen_model_label.pack(fill="none", side="top", expand=True)

        self.pred_complete_label = tkinter.Label(labels_frame, text="", 
                                                 anchor="center", 
                                                 background="gray25", 
                                                 foreground=self.primary_color)
        self.pred_complete_label.pack(fill="none", side="top", expand=True)

        extract_frames_button = tkinter.Button(options_frame, 
                                        text="Extract Frames",
                                        anchor="center", foreground="white",
                                        background=self.primary_color,
                                        activebackground=self.active_color,
                                        borderwidth=0, 
                                        command=self.extract_frames_from_video)
        
        extract_frames_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10,
                                side="left", expand=True)
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

        track_images_button = tkinter.Button(options_frame, text="Apply Tracking",
                                     anchor="center", 
                                     foreground="white",
                                     background=self.primary_color,
                                     activebackground=self.active_color,
                                     borderwidth=0,
                                     command=self.apply_tracking)
        track_images_button.pack(fill="none", padx=10, pady=10, ipadx=10,
                                ipady=10, side="left", expand=True)

        pred_img_button = tkinter.Button(options_frame, text="Predict Image",
                                         anchor="center", foreground="white",
                                         background=self.primary_color, 
                                         activebackground=self.active_color,
                                         borderwidth=0,
                                         command=self.choose_predictor)
        pred_img_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10, 
                             side="left", expand=True)

        # self.model_menu = tkinter.ttk.Combobox(options_frame, 
        #                                        values=self.model_options,
        #                                        background="gray90",
        #                                        state="readonly", width=12)
        # self.model_menu.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10,
        #                      side="left", expand=True)
        # self.model_menu.set(self.model_options[0])

        choose_weight_button = tkinter.Button(options_frame, 
                                              text="Choose Weight", 
                                              anchor="center", 
                                              foreground="white",
                                              background=self.primary_color,
                                              activebackground=self.active_color,
                                              border=0,
                                              command=self.choose_model_weight)
        choose_weight_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10, 
                             side="left", expand=True)

        '''
        copy_label_button = tkinter.Button(options_frame, 
                                           text="Copy Label",
                                           anchor="center", 
                                           foreground="white",
                                           background=self.primary_color,  
                                           activebackground=self.active_color,
                                           borderwidth=0,
                                           command=self.copy_label)
        copy_label_button.pack(fill="none", padx=10, pady=10, ipadx=10, 
                               ipady=10, side="left", expand=True)
        
        copy_label_button = tkinter.Button(options_frame, 
                                           text="Paste Label",
                                           anchor="center", 
                                           foreground="white",
                                           background=self.primary_color,  
                                           activebackground=self.active_color,
                                           borderwidth=0,
                                           command=self.paste_label)
        copy_label_button.pack(fill="none", padx=10, pady=10, ipadx=10, 
                               ipady=10, side="left", expand=True)
        '''
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

        self.canvas = tkinter.Canvas(self.parent, background="gray25", 
                                     borderwidth=0,
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

        self.frame_slider = tkinter.Scale(slider_frame, background="gray25",
                                          foreground="white", 
                                          orient="horizontal",
                                          state="disabled", border=0,
                                          highlightthickness=0,
                                          length=1000, label="Frame Number", 
                                          from_=1, to=700, relief="flat",
                                          variable=self.frame_num,
                                          command=self.slide_image)
        self.frame_slider.pack(fill="none", side="top", expand=True)

        add_label_cls_button = tkinter.Button(label_update_frame,
                                    text="New Class",
                                    anchor="center", 
                                    foreground="white",
                                    background=self.primary_color,
                                    activebackground=self.active_color,
                                    borderwidth=0,
                                    command=self.add_label_class)
        add_label_cls_button.pack(fill="none", padx=0, pady=10, ipadx=10,
                                  ipady=10, side="left", expand=True)
        
        add_label_cls_entry = tkinter.Entry(label_update_frame, bg="gray90",
                                            textvariable=self.new_label_cls)
        add_label_cls_entry.pack(fill="none", padx=20, pady=10, ipadx=10,
                                  ipady=10, side="left", expand=True)

        change_label_button = tkinter.Button(label_update_frame, 
                                             text="Change Class",
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

        change_label_id_button = tkinter.Button(label_update_frame, 
                                                text="Change ID",
                                                anchor="center", 
                                                foreground="white",
                                                background=self.warning_color,  
                                                activebackground=self.active_color,
                                                borderwidth=0, 
                                                command=self.update_labels)
        change_label_id_button.pack(fill="none", padx=0, pady=10, ipadx=10, 
                                    ipady=10, side="left", expand=True)
        
        label_id_entry = tkinter.Entry(label_update_frame, bg="gray90", 
                                       textvariable=self.label_id)
        label_id_entry.pack(fill="none", padx=20, pady=10, ipadx=10, 
                            ipady=10, side="left", expand=True)
        
        delete_label_button = tkinter.Button(label_update_frame, text="Delete Label",
                                anchor="center", foreground="white",
                                background=self.danger_color,
                                activebackground=self.active_color,
                                borderwidth=0, command=self.delete_selected_label)
        
        delete_label_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10,
                                side="left", expand=True)
        
        # Add Multiframe Edit button to the bottom_button_frame
        multiframe_edit_button = tkinter.Button(label_update_frame, text="Multiframe-Edit",
                                                anchor="center", foreground="white",
                                                background=self.primary_color,
                                                activebackground=self.active_color,
                                                borderwidth=0, command=self.open_multiframe_edit_window)
        multiframe_edit_button.pack(fill="none", padx=10, pady=10, ipadx=10, ipady=10,
                                    side="right")
        
        self.parent.bind("<KeyPress-Right>", self.get_arrow_keys)
        self.parent.bind("<KeyPress-Left>", self.get_arrow_keys)
        self.parent.bind("<Control-c>", self.get_keyboard_shortcut)
        self.parent.bind("<Control-v>", self.get_keyboard_shortcut)

        self.parent.bind("<w>", self.move_label_up)
        self.parent.bind("<a>", self.move_label_left)
        self.parent.bind("<s>", self.move_label_down)
        self.parent.bind("<d>", self.move_label_right)
        
        self.canvas.bind("<Configure>", self.resize_image)
        self.canvas.bind("<B1-Motion>", self.get_dragging_coords)
        self.canvas.bind("<ButtonRelease-1>", self.get_mouse_release_coords)
        self.canvas.bind("<Button-1>", self.get_mouse_click_coords)
        self.canvas.bind("<Button-3>", self.delete_labels)

    def open_multiframe_edit_window(self):
        """
        Creates a multiframe_window that lets users change or delete 
        labels in multiple frames with a single click.
        
        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        max_frame = len(self.images_list)
        multiframe_window = tkinter.Toplevel(self.parent)
        multiframe_window.title("Multiframe Edit")

        start_frame_label = tkinter.Label(multiframe_window, text="Start Frame:")
        start_frame_label.grid(row=0, column=0, padx=10, pady=5)
        self.start_frame_entry = tkinter.Entry(multiframe_window)
        self.start_frame_entry.grid(row=0, column=1, padx=10, pady=5)

        end_frame_label = tkinter.Label(multiframe_window, text="End Frame:")
        end_frame_label.grid(row=1, column=0, padx=10, pady=5)
        self.end_frame_entry = tkinter.Entry(multiframe_window)
        self.end_frame_entry.grid(row=1, column=1, padx=10, pady=5)

        label_id_label = tkinter.Label(multiframe_window, text="Label ID:")
        label_id_label.grid(row=2, column=0, padx=10, pady=5)
        self.label_id_entry = tkinter.Entry(multiframe_window)
        self.label_id_entry.grid(row=2, column=1, padx=10, pady=5)

        action_label = tkinter.Label(multiframe_window, text="Action:")
        action_label.grid(row=3, column=0, padx=10, pady=5)
        action_menu_values = ["Change ID", "Delete", "Change Class", "Swap ID",
                              "Copy Label"]
        self.action_menu = tkinter.ttk.Combobox(multiframe_window, 
                                                values=action_menu_values,
                                                state="readonly")
        self.action_menu.grid(row=3, column=1, padx=10, pady=5)
        self.action_menu.set("Change ID")

        new_value_label = tkinter.Label(multiframe_window, text="Value:")
        new_value_label.grid(row=4, column=0, padx=10, pady=5)
        self.new_value_entry = tkinter.Entry(multiframe_window)
        self.new_value_entry.grid(row=4, column=1, padx=10, pady=5)

        apply_button = tkinter.Button(multiframe_window, text="Apply", 
                                      command=self.apply_multiframe_edit)
        apply_button.grid(row=5, column=0, columnspan=2, pady=10)

    def apply_multiframe_edit(self):
        """
        Reads user choices from multiframe_window in 
        self.open_multiframe_edit_window function and applies the 
        command the user has chosen to frames between start frame and
        end frame chosen by user

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        return_image = self.chosen_image_name

        try:
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())
            label_id = int(self.label_id_entry.get())
            action = self.action_menu.get()
            new_value = self.new_value_entry.get()
            new_value = int(new_value) if new_value else None

            if start_frame > end_frame:
                messagebox.showerror(title="Invalid Input", 
                                     message="Start frame cannot be greater than end frame.")
                return

            if end_frame > len(self.images_list):
                messagebox.showerror(title="Invalid Input", 
                                     message=f"End frame cannot be greater than the total number of frames ({len(self.images_list)}).")
                return

        except ValueError:
            messagebox.showerror(title="Invalid Input", 
                                 message="Please enter valid input values.")
            return
        
        unchanged_frames = []
        is_label_found = False

        for frame_num in range(start_frame, end_frame + 1):
            image_name = self.images_list[frame_num - 1]
            self.chosen_image_name = image_name
            self.load_labels()

            if action == "Delete":
                self.pred_labels[1] = [label for label in self.pred_labels[1] if label[0] != label_id]
            elif action == "Change ID":
                new_id = new_value
                if any(label[0] == new_id for label in self.pred_labels[1]):
                    unchanged_frames.append(frame_num)
                    continue
                for label in self.pred_labels[1]:
                    if label[0] == label_id:
                        label[0] = new_id
            elif action == "Change Class":
                new_class = new_value
                for label in self.pred_labels[1]:
                    if label[0] == label_id:
                        label[1] = new_class
            elif action == "Swap ID":
                if new_value is None:
                    messagebox.showerror(title="Invalid Input", 
                                         message="Please enter a valid ID to swap.")
                    return
                for label in self.pred_labels[1]:
                    if label[0] == label_id:
                        label[0] = new_value
                    elif label[0] == new_value:
                        label[0] = label_id
            elif action == "Copy Label":
                # Scanning frames just to find label to copy
                for label in self.pred_labels[1]:
                    if label[0] == label_id:
                        self.copied_label = label
                        is_label_found = True

            if action != "Copy Label":
                self.save_preds_to_disk()

        if action == "Copy Label" and is_label_found:
            # Iterating through frames a second time to paste label
            for frame_num in range(start_frame, end_frame + 1):
                image_name = self.images_list[frame_num - 1]
                self.chosen_image_name = image_name
                self.load_labels()

                if self.copied_label not in self.pred_labels[1]:
                    self.pred_labels[1].append(self.copied_label)
                    self.save_preds_to_disk()
        elif action == "Copy Label" and not is_label_found:
            msg = "Couldn't find label to copy within frames from "
            msg += f"{start_frame} to {end_frame}"
            messagebox.showwarning(title="Label Not Found", 
                                   message=msg)
            
            return

        # Reload original frame labels
        self.chosen_image_name = return_image
        self.load_labels()
        self.draw_labels()

        if unchanged_frames:
            messagebox.showwarning(title="Unchanged Frames", 
                                   message=f"Some frames were not changed due to ID conflicts: {unchanged_frames}")
        else:
            messagebox.showinfo(title="Success", 
                                message="Multiframe edit applied successfully.")

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

            # Used to set upper limit of self.frame_slider
            images_count = len(self.images_list)
            
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
                msg += "Do you want to use the existing directory or "
                msg += "overwrite it?"
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

            self.label_options_txt_dir = os.path.join(self.labels_folder,
                                                      'label_options.txt')
            self.label_colors_txt_dir = os.path.join(self.labels_folder,
                                                      'label_colors.txt')
            
            # Creating a txt file to hold initial label options in it 
            # and update when user adds new labels
            # If label_options.txt already exists, user has opened this
            # folder before, and maybe added their own label classes
            if not os.path.exists(self.label_options_txt_dir):
                with open(self.label_options_txt_dir, "+w") as label_options_txt:
                    label_options_txt.write("\n".join(self.label_options))
            else:
                # If label_options.txt already exists, load its content
                # into self.label_options
                with open(self.label_options_txt_dir) as label_options_txt:
                    self.label_options = label_options_txt.read().splitlines()

                self.label_change_menu.config(values=self.label_options)

            # Creating a txt file to hold initial label options in it 
            # and update when user adds new labels
            # If label_colors.txt already exists, user has opened this
            # folder before, and maybe added their own label colors
            if not os.path.exists(self.label_colors_txt_dir):
                with open(self.label_colors_txt_dir, "+w") as label_colors_txt:
                    label_colors_txt.write("\n".join(self.label_colors))
            else:
                # If label_options.txt already exists, load its content
                # into self.label_options
                with open(self.label_colors_txt_dir) as label_colors_txt:
                    self.label_colors = label_colors_txt.read().splitlines()

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
            resized_cv_image = cv2.resize(self.cv_image, canvas_size)
                
            # Written self.pil_image to prevent Garbage Collector from
            # deleting function scope image
            # Read "Displaying Image In Tkinter Python" article in C# 
            # Corner website for more info, link in "Resources Used" 
            # at the top
            self.pil_image = ImageTk.PhotoImage(Image.fromarray(resized_cv_image))
                    
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", 
                                                            image=self.pil_image,
                                                            tag="canvas_image")
            
            # Load labels for the chosen image
            self.load_labels()

            if self.pred_labels != [] and self.pred_labels[1] != []:
                self.draw_labels()
        else:
            self.chosen_img_label.config(text=f"Chosen Image: ")
            # self.canvas.r  # self.canvas.r ?

        self.frame_slider.config(state="normal", to=images_count)

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
        
        #print("Current frame labels: ", self.pred_labels)
        #print("Previous frame labels: ", self.previous_frame_pred_labels)
        #print("=" * 52)
            
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
            # Calling self.save_preds_to_disk() when there are labels to
            # save to decrease the amount of times this function
            # gets called and hopefully solve RecursionError
            if self.pred_labels != [] and self.pred_labels[1] != []:
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

            self.frame_slider.set(file_index - i + 1)

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
                resized_cv_image = cv2.resize(self.cv_image, canvas_size)

                # Written self.pil_image to prevent Garbage Collector from
                # deleting function scope image
                # Read "Displaying Image In Tkinter Python" article in C# 
                # Corner website for more info, link in "Resources Used" 
                # at the top
                self.pil_image = ImageTk.PhotoImage(Image.fromarray(resized_cv_image))

                self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw",
                                                                image=self.pil_image,
                                                                tag="canvas_image")
                
                # To make clear chosen labels don't carry between frames
                self.selected_box_label.config(text=f"Selected Box: ")

                # Load labels for the chosen image
                self.load_labels()
                
                # Calling self.draw_labels() when there are labels to
                # draw to decrease the amount of times this function
                # gets called and hopefully solve RecursionError
                if self.pred_labels != [] and self.pred_labels[1] != []:
                    self.draw_labels()
                
            else:
                self.chosen_img_label.config(text=f"Chosen Image: ")
                    
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
            # Calling self.save_preds_to_disk() when there are labels to
            # save to decrease the amount of times this function
            # gets called and hopefully solve RecursionError
            if self.pred_labels != [] and self.pred_labels[1] != []:
                self.save_preds_to_disk()  # Automatically save labels

            self.pred_complete_label.config(text="")

            i = 1
            self.chosen_image_name = self.images_list[file_index + i]
            file_dir = os.path.join(self.folder_name, self.chosen_image_name)

            while (i < len(self.images_list) - file_index and 
                    not os.path.exists(file_dir)):
                i += 1
                self.chosen_image_name = self.images_list[file_index + i]
                file_dir = os.path.join(self.folder_name,
                                        self.chosen_image_name)
                
            self.frame_slider.set(file_index + i + 1)

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
                resized_cv_image = cv2.resize(self.cv_image, canvas_size)

                # Written self.pil_image to prevent Garbage Collector from
                # deleting function scope image
                # Read "Displaying Image In Tkinter Python" article in C# 
                # Corner website for more info, link in "Resources Used" 
                # at the top
                self.pil_image = ImageTk.PhotoImage(Image.fromarray(resized_cv_image))

                self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw",
                                                                image=self.pil_image,
                                                                tag="canvas_image")
                
                # To make clear chosen labels don't carry between frames
                self.selected_box_label.config(text=f"Selected Box: ")

                # Load labels for the chosen image
                self.load_labels()

                # Calling self.draw_labels() when there are labels to
                # draw to decrease the amount of times this function
                # gets called and hopefully solve RecursionError
                if self.pred_labels != [] and self.pred_labels[1] != []:
                    self.draw_labels()
            else:
                self.chosen_img_label.config(text=f"Chosen Image: ")

    def slide_image(self, scale):
        """
        Gets last scale value from self.frame_slider, uses it to open 
        image with scale index in self.images_list

        Parameters
        ----------
        scale : float
            Shows chosen value of self.frame_slider

        Returns
        ----------
        None 
        """
        
        # This if-else is to prevent IndexError 
        # when scale = len(self.images_list)
        if int(scale) < len(self.images_list):
            self.chosen_image_name = self.images_list[int(scale)-1]
        else:
            self.chosen_image_name = self.images_list[-1]

        # self.save_preds_to_disk()

        file_dir = os.path.join(self.folder_name, self.chosen_image_name)

        # Resetting self.cv_image if an image has been opened before,
        # to prevent drawing labels on top of label drawn image
        if self.cv_image is not None:
            self.cv_image = None

        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        
        if file_dir != "":
            self.chosen_img_label.config(text=f"Chosen Image: {file_dir}")

            self.cv_image = cv2.imread(file_dir)
            
            resized_cv_image = cv2.resize(self.cv_image, canvas_size)
                
            # Written self.pil_image to prevent Garbage Collector from
            # deleting function scope image
            # Read "Displaying Image In Tkinter Python" article in C# 
            # Corner website for more info, link in "Resources Used" 
            # at the top
            self.pil_image = ImageTk.PhotoImage(Image.fromarray(resized_cv_image))
                    
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", 
                                                            image=self.pil_image,
                                                            tag="canvas_image")
            
            # Calling self.load_labels() and self.draw_labels() when 
            # there are labels to draw to decrease the amount of times 
            # these functions get called and hopefully solve 
            # RecursionError
            if self.pred_labels != [] and self.pred_labels[1] != []:
                # Load labels for the chosen image
                self.load_labels()
                self.draw_labels()
        else:
            self.chosen_img_label.config(text=f"Chosen Image: ")

    def resize_image(self, event):
        """
        Reads events received from widget it's binded to (self.canvas),
        if event is user resizing window (and canvas with it), resizes
        image to fit new canvas. Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user input this function handles

        Returns
        ----------
        None
        """
        
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

    def get_keyboard_shortcut(self, event):
        """
        Reads events from the widget it's binded to (self.parent), if
        event is user pressing Control+C, calls self.copy_label() 
        function; if event is user pressing Control+V, calls 
        self.paste_label() function. Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user event this function handles

        Returns
        ----------
        None
        """
        
        if event.state & 0x0004 == 0x0004 and event.keycode == 67:
            self.copy_label()
        elif event.state & 0x0004 == 0x0004 and event.keycode == 86:
            self.paste_label()


    def move_label_up(self, event):
        """
        Reads events from the widget it's binded to (self.parent), if
        event is user pressing W button, moves the self.selected_label 
        up by decreasing its y_center value by a certain number, calls 
        self.update_pred_labels() and self.draw_labels() functions to 
        update self.pred_labels list and display moved label to user. 
        Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user event this function handles

        Returns
        ----------
        None
        """

        if self.selected_label:
            # Move up by decreasing y_center
            self.selected_label[3] = max(0, self.selected_label[3] - 0.005)
            self.update_pred_labels(self.selected_label)
            self.draw_labels()

    def move_label_down(self, event):
        """
        Reads events from the widget it's binded to (self.parent), if
        event is user pressing S button, moves the self.selected_label 
        down by increasing its y_center value by a certain number, calls 
        self.update_pred_labels() and self.draw_labels() functions to 
        update self.pred_labels list and display moved label to user. 
        Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user event this function handles

        Returns
        ----------
        None
        """

        if self.selected_label:
            # Move down by increasing y_center
            self.selected_label[3] = min(1, self.selected_label[3] + 0.005)
            self.update_pred_labels(self.selected_label)
            self.draw_labels()

    def move_label_left(self, event):
        """
        Reads events from the widget it's binded to (self.parent), if
        event is user pressing A button, moves the self.selected_label 
        left by decreasing its x_center value by a certain number, calls 
        self.update_pred_labels() and self.draw_labels() functions to 
        update self.pred_labels list and display moved label to user. 
        Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user event this function handles

        Returns
        ----------
        None
        """
        
        if self.selected_label:
            # Move left by decreasing x_center
            self.selected_label[2] = max(0, self.selected_label[2] - 0.005)  
            self.update_pred_labels(self.selected_label)
            self.draw_labels()

    def move_label_right(self, event):
        """
        Reads events from the widget it's binded to (self.parent), if
        event is user pressing D button, moves the self.selected_label 
        right by increasing its x_center value by a certain number, 
        calls self.update_pred_labels() and self.draw_labels() functions 
        to update self.pred_labels list and display moved label to user. 
        Does nothing for other user inputs.

        Parameters
        ----------
        event : tkinter event
            The user event this function handles

        Returns
        ----------
        None
        """
        
        if self.selected_label:
            # Move right by increasing x_center
            self.selected_label[2] = min(1, self.selected_label[2] + 0.005)  
            self.update_pred_labels(self.selected_label)
            self.draw_labels()

    def update_pred_labels(self, updated_label):
        """
        Finds the label in self.pred_labels that matches ID of 
        updated_label and assigns updated_label to the matching label

        Parameters
        ----------
        updated_label : list
            Holds a single label in a frame

        Returns
        ----------
        None
        """
        
        for i, label in enumerate(self.pred_labels[1]):
            if label[0] == updated_label[0]:
                self.pred_labels[1][i] = updated_label
                break

    def copy_label(self):
        """
        Lets user choose and copy a label from a frame into 
        self.copied_label

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        click_x = self.chosen_label_x
        click_y = self.chosen_label_y
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
    
        # Cleaning self.copied_label to ensure it only holds latest
        # copied label
        if self.copied_label != []:
            self.copied_label = []
    
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
                    self.copied_label.append(label)
    
                    break

    def paste_label(self):
        """
        Lets users paste a label from self.copied_label to a frame

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.pred_labels != [] and self.pred_labels[1] != []:
            self.pred_labels[1].append(self.copied_label[0])
        elif self.pred_labels != [] and self.pred_labels[1] == []:
            # Assuming self.copied_label has a single label in it
            self.pred_labels[1].append(self.copied_label[0])
        else:
            # Assuming self.copied_label has a single label in it
            self.pred_labels.append(self.chosen_image_name)
            self.pred_labels.append([self.copied_label[0]])
            
        self.draw_labels()

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

    def add_label_class(self):
        """
        Lets user add new label classes to self.label_options list and
        lets user assign an unused color to self.label_colors

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        if self.new_label_cls != "":
            new_label_cls = self.new_label_cls.get()

            # label_color = ((red, green, blue), '#hexcode')
            label_color = colorchooser.askcolor(title="New Label's Color")

            # Prevents users from choosing an already chosen color by
            # looping until user chooses a color that is not inside 
            # self.label_colors
            while label_color[1] in self.label_colors:
                msg = "You have chosen a color which is already in use. "
                msg += "Please choose a different color.\n"
                msg += f"Current colors: {self.label_colors}"

                messagebox.showerror(title="Color Is Already Used",
                                     message=msg)
                
                label_color = colorchooser.askcolor(title="New Label's Color")
                print(label_color)

            self.label_options.append(new_label_cls)
            self.label_colors.append(label_color[1])

            with open(self.label_options_txt_dir, '+a') as label_options_txt:
                label_options_txt.write(f"\n{new_label_cls}")

            with open(self.label_colors_txt_dir, '+a') as label_colors_txt:
                label_colors_txt.write(f"\n{label_color[1]}")

            self.new_label_cls.set("")

            self.label_change_menu.config(values=self.label_options)

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
        
        # chosen_model = self.model_menu.get()
        self.pred_labels = []

        if self.chosen_model == "yolov8s":
            self.predict_with_yolo()
        elif self.chosen_model == "faster_rcnn":
            self.predict_with_faster_rcnn()
        elif self.chosen_model == "retina_net":
            self.predict_with_retina_net()
        else:
            messagebox.showerror(message="Unknown predictor!!!", 
                                 title="Unknown Choice")

    def choose_model_weight(self):
        """
        Opens a Open File dialog and lets user choose a weight for the
        model they have chosen with the self.model_menu

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        filetypes_list = [["PyTorch Weights", "*.pth"], 
                            ["PyTorch Weights", "*.pt"]]
    
        weight_dir = filedialog.askopenfilename(filetypes=filetypes_list, 
                                            initialdir=self.script_dir, 
                                            title="Choose Weight File")
        
        weight_name = os.path.split(weight_dir)[1]

        # Turning weight_name to lowercase to make sure we don't
        # miss variations of 'yolo', 'best', 'last', 'faster',
        # 'rcnn', 'r-cnn', 'retina'
        if (weight_name.lower().find("yolo") != -1 
                or weight_name.lower().find("best") != -1 
                or weight_name.lower().find("last") != -1):
            self.yolo_weights_dir = weight_dir

            # self.model_menu.set("yolov8s")
            self.chosen_model = "yolov8s"
            self.chosen_model_label.config(text="Chosen Model: YOLOv8s")
        elif (weight_name.lower().find("faster") != -1 
                and (weight_name.lower().find("rcnn") != -1
                or weight_name.lower().find("r-cnn") != -1)):
            self.faster_rcnn_weights_dir = weight_dir

            # self.model_menu.set("faster_rcnn")
            self.chosen_model = "faster_rcnn"
            self.chosen_model_label.config(text="Chosen Model: Faster R-CNN")
        elif weight_name.lower().find("retina") != -1:
            self.retinanet_weights_dir = weight_dir

            # self.model_menu.set("retina_net")
            self.chosen_model = "retina_net"
            self.chosen_model_label.config(text="Chosen Model: RetinaNet")
        else:
            msg = "Please make sure you have chosen a suitable model!\n"
            msg += "You can choose a model from the dropdown menu if you want."

            messagebox.showerror(title="Unknown Choice", message=msg)            
            self.chosen_model = ""
            self.chosen_model_label.config(text="Chosen Model:")

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

        if self.sort_tracking_enabled == False:
            self.assign_ids(new_predictions) 
        else:
            self.pred_labels = [self.chosen_image_name, new_predictions]    

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
        if self.sort_tracking_enabled == False:
            self.assign_ids(new_predictions) 
        else:
            self.pred_labels = [self.chosen_image_name, new_predictions]    

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

        if self.sort_tracking_enabled == False:
            self.assign_ids(new_predictions) 
        else:
            self.pred_labels = [self.chosen_image_name, new_predictions]    

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
        
        if (self.pred_labels != [] and self.pred_labels[1] != [] and
              self.chosen_image_name == self.pred_labels[0]):
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

                rect_color = self.label_colors[pred_class]

                # Check if this label is the selected one
                if self.selected_label and self.selected_label[0] == counter:
                    rect_color = "red"
                    
    
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

    def delete_selected_label(self):
        if self.cv_image is None:
            messagebox.showerror(title="No Image Selected",
                                message="No image selected to delete labels from.")
            return

        if self.selected_label is None:
            messagebox.showerror(title="No Label Selected",
                                message="No label selected to delete.")
            return

        if self.pred_labels != []:
            self.pred_labels[1].remove(self.selected_label)
            self.canvas.delete(f"rectangle_{self.selected_label[0]}")
            self.canvas.delete(f"text_{self.selected_label[0]}")
            self.selected_label = None
            self.selected_box_label.config(text="Selected Box: ")
            self.draw_labels()

    def update_labels(self):
        if self.cv_image is None:
            messagebox.showerror(title="No Image Selected", message="No image selected to update labels of.")
            return

        if self.chosen_label_x is None or self.chosen_label_y is None:
            messagebox.showerror(title="No Label Selected", message="No label selected to update.")
            return

        click_x = self.chosen_label_x
        click_y = self.chosen_label_y

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        previous_selected_label = self.selected_label  # Store previous selected label
        self.selected_label = None  # Reset selected_label

        if self.pred_labels != []:
            for label in self.pred_labels[1]:
                width = int(label[4] * canvas_width)
                height = int(label[5] * canvas_height)
                x0 = int(label[2] * canvas_width) - width // 2
                x1 = int(label[2] * canvas_width) + width // 2
                y0 = int(label[3] * canvas_height) - height // 2
                y1 = int(label[3] * canvas_height) + height // 2

                is_in_x = (click_x - x0 - 4) < width and (x1 - click_x - 4) < width
                is_in_y = (click_y - y0 - 4) < height and (y1 - click_y - 4) < height

                if is_in_x and is_in_y:
                    self.selected_box_label.config(text=f"Selected Box: {label[0]}")
                    self.selected_label = label  # Store selected label

                    try:
                        chosen_label = self.label_change_menu.get()
                        label_index = self.label_options.index(chosen_label)

                        if label_index >= 1:
                            label[1] = label_index - 1

                        self.label_change_menu.set(self.label_options[0])
                    except ValueError as e:
                        messagebox.showerror(title="Error While Choosing Label", message=f"Error: {e}")

                    if self.label_id.get() != "":
                        i = 0
                        label_id = int(self.label_id.get())

                        self.selected_box_label.config(text=f"Selected Box: {label[0]}")

                        while (label_id != self.pred_labels[1][i][0] and i < len(self.pred_labels[1]) - 1):
                            i += 1

                        if (i >= len(self.pred_labels[1]) - 1 and label_id != self.pred_labels[1][-1][0]):
                            self.canvas.delete(f"text_{label[0]}")
                            label[0] = label_id

                        self.selected_box_label.config(text=f"Selected Box: {label[0]}")

                        self.label_id.set("")
                        self.label_change_menu.set(self.label_options[0])

                    break

        # Update the color of the previous selected label to its original color
        if previous_selected_label:
            print("Previous selected label condition")
            print(previous_selected_label)
            self.canvas.itemconfig(f"rectangle_{previous_selected_label[0]}", outline=self.label_colors[previous_selected_label[1]])
            self.canvas.itemconfig(f"text_{previous_selected_label[0]}", fill=self.label_colors[previous_selected_label[1]])

        # Update the color of the new selected label to red
        if self.selected_label:
            print("self.selected_label condition")
            print(self.selected_label)
            self.canvas.itemconfig(f"rectangle_{self.selected_label[0]}", outline="red")
            self.canvas.itemconfig(f"text_{self.selected_label[0]}", fill="red")
        self.parent.update_idletasks()  # Ensure the canvas is updated immediately
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

        # Using str.split instead of os.path.splitext hoping to solve
        # RecursionError related to this line
        # base_filename = os.path.splitext(self.chosen_image_name)[0]
        base_filename = str.split(self.chosen_image_name, sep=".")[0]

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
                label_id = pred_label[0]
                pred_class = pred_label[1]
                x_middle = pred_label[2]
                y_middle = pred_label[3]
                box_width = pred_label[4]
                box_height = pred_label[5]

                if is_yolo_format:
                    labels_str += f"{label_id} {pred_class} {x_middle} "
                    labels_str += f"{y_middle} {box_width} {box_height}\n"
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

                    labels_str += f"{label_id} {pred_class} {x_start} "
                    labels_str += f"{y_start} {x_end} {y_end}\n"

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
                max_id = max([label[0] for label in self.previous_frame_pred_labels[1]], default=0)
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

    def extract_frames_from_video(self):
        video_path = filedialog.askopenfilename(title="Select a Video File",
                                                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not video_path:
            messagebox.showwarning("No File Selected", "Please select a video file.")
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(os.path.dirname(video_path), video_name)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_filename = os.path.join(output_folder, f"{video_name}_{frame_count}.jpg")
            print(f"Frame {frame_count} has been read succesfully.")
            cv2.imwrite(frame_filename, frame)

        cap.release()
        messagebox.showinfo("Success", f"Extracted {frame_count} frames to folder: {output_folder}")

    def apply_tracking(self):
        self.stop_tracking = False  # Add a flag to control the tracking process

        def restart_application():
            python = sys.executable
            os.execl(python, python, *sys.argv)

        def tracking_task():
            tracker = Tracker()
            max_track_id = 0  # Variable to store the maximum track ID encountered

            # Check if labels folder exists and prompt the user
            if os.path.exists(self.labels_folder):
                proceed = messagebox.askyesno(title="Overwrite Labels Folder",
                                            message="The labels folder already exists and will be overwritten. Do you want to proceed?")
                if not proceed:
                    self.sort_tracking_enabled = False
                    return

            self.sort_tracking_enabled = True

            # Disable the main window to prevent user interaction
            self.parent.attributes('-disabled', True)

            # Create progress bar
            progress_window = tkinter.Toplevel(self.parent)
            progress_window.title("Tracking Progress")

            def on_closing():
                self.stop_tracking = True
                restart_application()

            progress_window.protocol("WM_DELETE_WINDOW", on_closing)

            progress_label = tkinter.Label(progress_window, 
                                           text="Processing images...")
            progress_label.pack(pady=10)
            progress_bar = tkinter.ttk.Progressbar(progress_window, 
                                                   orient="horizontal", 
                                                   length=300, 
                                                   mode="determinate")
            progress_bar.pack(pady=10)
            progress_bar["maximum"] = len(self.images_list)
            progress_text = tkinter.Label(progress_window, 
                                          text="0% (0 / {total})".format(total=len(self.images_list)))
            progress_text.pack(pady=10)

            frame_count = 0
            for idx, image_name in enumerate(self.images_list):
                frame_count += 1
                if self.stop_tracking:  # Check if the stop flag is set
                    break

                image_path = os.path.join(self.folder_name, image_name)
                frame = cv2.imread(image_path)
                self.cv_image = frame
                self.chosen_image_name = image_name
                self.choose_predictor()

                if self.pred_labels == []:
                    msg = "No prediction labels found. "
                    msg += "Are you sure you have chosen a model for tracking?"
                    messagebox.showerror(title="Error While Tracking",
                                         message=msg)
                    
                    # Enable the main window so users can try again
                    self.parent.attributes('-disabled', False)
                    
                    return

                detections = []
                if self.pred_labels[1] != []:
                    for pred_label in self.pred_labels[1]:
                        x_center = pred_label[1]
                        y_center = pred_label[2]
                        width = pred_label[3]
                        height = pred_label[4]
                        label_cls = int(pred_label[0])
                        detections.append((x_center, y_center, width, height, label_cls))

                    tracker_predictions = tracker.update(detections)
                    if tracker_predictions:
                        new_pred_labels = []
                        for pred in tracker_predictions:
                            id = int(pred[0])
                            if id > max_track_id:  # Update max_track_id if a higher ID is encountered
                                max_track_id = id
                            x_center = float(pred[1])
                            y_center = float(pred[2])
                            width = float(pred[3])
                            height = float(pred[4])
                            label_cls = int(pred[5])
                            tracker_age = pred[6]
                            if tracker_age > 3 or frame_count <= 10:
                                new_pred_labels.append([id, label_cls, x_center, y_center, width, height])

                            

                        self.pred_labels[1] = new_pred_labels
                        self.save_preds_to_disk()

                        # Write the max_track_id to a file after each iteration
                        with open(os.path.join(self.labels_folder, "max_track_id.txt"), "w") as f:
                            f.write(str(max_track_id))

                # Update progress bar
                progress_bar["value"] = idx + 1
                progress_percentage = int(((idx + 1) / len(self.images_list)) * 100)
                progress_text.config(text="{percent}% ({current} / {total})".format(percent=progress_percentage, current=idx + 1, total=len(self.images_list)))
                progress_window.update_idletasks()

            self.sort_tracking_enabled = False
            progress_window.destroy()
            # Re-enable the main window
            self.parent.attributes('-disabled', False)
            if not self.stop_tracking:
                messagebox.showinfo(title="Tracking Complete", message="Tracking on the image folder is complete.")
            else:
                restart_application()

        tracking_thread = threading.Thread(target=tracking_task)
        tracking_thread.start()

            

            

if __name__ == "__main__":
    nn_gui = NeuralNetworkGUI()

    nn_gui.parent.mainloop()