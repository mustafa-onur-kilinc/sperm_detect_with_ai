"""
This GUI application is used to load images from folders, give images to
neural network models as input and show output of models to users.

WARNING: If you get locale.Error when trying to run this code, you need
to set locale in ttkbootstrap/dialogs/dialogs.py, line 566.
Check this link for details:
<https://github.com/israel-dryer/ttkbootstrap/issues/505#issuecomment-1791564851>

In order to work properly, this script needs a folder structure like 
this:
- main_dir
    - nn_gui.py
    - weights_dir
        - yolo_weight_dirname
            - yolo_weight_name
        - faster_rcnn_weight_dirname
            - faster_rcnn_weight_name
        - retina_net_weight_dirname
            - retina_net_weight_name
    - config_dirname
        - yaml_name

Resources used to write this script:
DataThinkers 2023, "GUI-for-Machine-Learning-Projects-Using-Tkinter", 
accessed 30 April 2024,
<https://github.com/DataThinkers/GUI-for-Machine-Learning-Projects-Using-Tkinter/tree/main>

tkinter.filedialog — File selection dialogs n.d.,
Python Software Foundation, accessed 30 April 2024,
<https://docs.python.org/3.11/library/dialog.html#module-tkinter.filedialog>

Sourav De 2023, "How to make your Python GUI look awesome", Medium,
accessed 30 April 2024,
<https://medium.com/@SrvZ/how-to-make-your-python-gui-look-awesome-9372c42d7df4>

Israel Dryer n.d., "ttkbootstrap", accessed 30 April 2024,
<https://ttkbootstrap.readthedocs.io/en/latest/>

PyYAML Documentation n.d., accessed 30 April 2024,
<https://pyyaml.org/wiki/PyYAMLDocumentation>

Borin --help 2021, "How to convert from tensor to float", 
Stack Exchange, Inc., accessed 30 April 2024, 
<https://stackoverflow.com/a/70043741>

tkinter — Python interface to Tcl/Tk n.d., Python Software Foundation, 
accessed 1 May 2024, <https://docs.python.org/3.11/library/tkinter.html#>

putText() n.d., OpenCV, accessed 1 May 2024,
<https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576>

Israel Dryer n.d., "MessageBox", accessed 3 May 2024,
<https://ttkbootstrap.readthedocs.io/en/latest/api/dialogs/messagebox/#ttkbootstrap.dialogs.dialogs.Messagebox>

rohanagarwal883 2021, "Python Tkinter – Entry Widget", GeeksforGeeks, 
accessed 3 May 2024,
<https://www.geeksforgeeks.org/python-tkinter-entry-widget/>

GeeksforGeeks 2023, "How to Remove an Item from the List in Python",
GeeksforGeeks, accessed 3 May 2024,
<https://www.geeksforgeeks.org/how-to-remove-an-item-from-the-list-in-python/>

rogeriopvl 2009, "How do I get time of a Python program's execution?",
Stack Exchange, Inc., accessed 11 May 2024,
<https://stackoverflow.com/a/1557584>

numpydoc maintainers n.d., "Style guide", accessed 3 May 2024,
<https://numpydoc.readthedocs.io/en/latest/format.html>

abhigoya 2020, "Dropdown Menus – Tkinter",
GeeksforGeeks, accessed 11 May 2024,
<https://www.geeksforgeeks.org/dropdown-menus-tkinter/>

Özgün Zeki BOZKURT's detect.py and get_model.py scripts,
received 7 May 2024.

Nihal Murmu 2023, "Displaying Image In Tkinter Python", C# Corner,
accessed 13 May 2024,
<https://www.c-sharpcorner.com/blogs/basics-for-displaying-image-in-tkinter-python>

Python Tkinter text editor: Save text to file 2023, w3resource.com,
accessed 11 May 2024,
<https://www.w3resource.com/python-exercises/tkinter/python-tkinter-dialogs-and-file-handling-exercise-10.php>

Kumar_Satyam 2020, "How to Get the Tkinter Label Text?", GeeksforGeeks, 
accessed 13 May 2024,
<https://www.geeksforgeeks.org/how-to-get-the-tkinter-label-text/>

Israel Dryer n.d., "Combobox", accessed 14 May 2024,
<https://ttkbootstrap.readthedocs.io/en/latest/styleguide/combobox/>
"""

import os
import cv2
import yaml
import json
import torch
import ttkbootstrap as ttk
import tkinter as tk

from tkinter import filedialog 
from ttkbootstrap.dialogs.dialogs import Messagebox
from ttkbootstrap.constants import *

from PIL import Image, ImageTk
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial

from ultralytics import YOLO

class NeuralNetworkGUI():
    """
    This class initializes a ttk window for GUI, lets user choose an 
    image to perform predictions on, loads pretrained weights for YOLO, 
    performs predictions using YOLO, shows label added image in a 
    separate window, lets user add or remove labels from image by using 
    GUI
    """
    
    def __init__(self):
        config_dirname = "config"
        yaml_name = "nn_gui_config.yaml"
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

        gui_theme = args_dict["theme"]
        gui_title = args_dict["title"]
        gui_geometry = args_dict["geometry"]
        self.model_options = args_dict["model_options"]
        self.save_options = args_dict["save_options"]

        self.root = ttk.Window(themename=gui_theme)
        self.root.title(gui_title)
        self.root.geometry(gui_geometry)

        # Cheap solution to avoid having to dynamically resize widgets
        self.root.resizable(False, False)

        self.label_num_del = tk.StringVar()
        self.label_x_start = tk.StringVar()
        self.label_y_start = tk.StringVar()
        self.label_x_end = tk.StringVar()
        self.label_y_end = tk.StringVar()
        self.label_class = tk.StringVar()
        self.chosen_model = tk.StringVar()
        self.chosen_save_type = tk.StringVar()

        self.chosen_model.set(self.model_options[0])
        self.chosen_save_type.set(self.save_options[0])

        self.init_window()

        self.cv_image = None
        self.pred_labels = []
        self.pred_label_ids = []  # A list to easily check label_ids
        self.original_pred_labels_len = None

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
        
        script_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(script_dir, weights_dirname)
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
        Initializes a ttk Window

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        frame = ttk.Frame(master=self.root, padding=100)
        frame.grid(columnspan=8, rowspan=10)

        self.chosen_img_label = ttk.Label(master=self.root, text="Chosen Image:",
                                          bootstyle="primary")
        self.chosen_img_label.grid(column=0, columnspan=5, row=0)

        self.pred_complete_label = ttk.Label(master=self.root, text="", 
                                             bootstyle="primary")
        self.pred_complete_label.grid(column=0, columnspan=5, row=1)

        choose_img_button = ttk.Button(master=self.root, text="Choose Image", 
                                       width=15, bootstyle="primary", 
                                       command=self.open_image) 
        choose_img_button.grid(column=0, row=2, padx=15, pady=15)
        
        draw_labels_button = ttk.Button(master=self.root, text="Draw Labels", width=15,
                             bootstyle="primary",
                             command=self.draw_labels)
        draw_labels_button.grid(column=3, row=2, padx=15, pady=15)
            
        close_button = ttk.Button(master=self.root, text="Close", width=10,
                                  bootstyle="danger", 
                                  command=self.close_window)
        close_button.grid(column=7, row=2, padx=15, pady=15)

        pred_img_button = ttk.Button(master=self.root, text="Predict Image", 
                                     width=15, bootstyle="primary", 
                                     command=self.choose_predictor)
        pred_img_button.grid(column=1, row=2, padx=15, pady=15)

        model_menu = ttk.Combobox(self.root, values=self.model_options,
                                  bootstyle="primary", width=12)
        model_menu.configure(state="readonly")  # Prevents invalid entry
        model_menu.grid(column=2, row=2, padx=15, pady=15)

        save_labels_button = ttk.Button(master=self.root, text="Save Labels",
                                        width=15, bootstyle="primary",
                                        command=self.choose_save_method)
        save_labels_button.grid(column=4, row=2, padx=15, pady=15)

        save_type_menu = ttk.Combobox(self.root, values=self.save_options,
                                      bootstyle="primary", width=10)
        
        # Prevents invalid entry
        save_type_menu.configure(state="readonly")
        save_type_menu.grid(column=5, row=2, padx=15, pady=15)

        self.canvas1 = ttk.Canvas(master=self.root, width=768, height=432)
        self.canvas1.grid(column=0, row=3, columnspan=5, rowspan=6)

        self.id_to_del_text = "Label ID to Delete:"
        id_to_del_label = ttk.Label(master=self.root, 
                                    text=self.id_to_del_text, 
                                    justify="left", bootstyle="default")
        id_to_del_label.grid(column=5, row=3, padx=0, pady=0)

        del_label_button = ttk.Button(master=self.root, text="Delete Label", 
                                      width=12, bootstyle="warning",
                                      command=self.delete_labels)
        del_label_button.grid(column=7, row=3, padx=15, pady=15)

        del_label_entry = tk.Entry(master=self.root, bg="white", 
                                   textvariable=self.label_num_del)
        del_label_entry.grid(column=6, row=3, padx=15, pady=15)

        add_label_button = ttk.Button(master=self.root, text="Add Label", 
                                      width=15, bootstyle="success",
                                      command=self.add_labels)
        add_label_button.grid(column=5, row=9, padx=15, pady=15)

        self.x_start_text = "X_start:"
        x_start_label = ttk.Label(master=self.root, 
                                  text=self.x_start_text, 
                                  justify="left", bootstyle="default")
        x_start_label.grid(column=5, row=4, padx=0, pady=0)

        x_start_entry = tk.Entry(master=self.root, bg="white", 
                                 textvariable=self.label_x_start)
        x_start_entry.grid(column=6, row=4, padx=15, pady=15)
 
        self.y_start_text = "Y_start:"
        y_start_label = ttk.Label(master=self.root, 
                                  text=self.y_start_text, 
                                  justify="left", bootstyle="default")
        y_start_label.grid(column=5, row=5, padx=0, pady=0)

        y_start_entry = tk.Entry(master=self.root, bg="white", 
                                 textvariable=self.label_y_start)
        y_start_entry.grid(column=6, row=5, padx=15, pady=15)

        self.x_end_text = "X_end:"
        x_end_label = ttk.Label(master=self.root, text=self.x_end_text, 
                                justify="left", bootstyle="default")
        x_end_label.grid(column=5, row=6, padx=0, pady=0)

        x_end_entry = tk.Entry(master=self.root, bg="white", 
                               textvariable=self.label_x_end)
        x_end_entry.grid(column=6, row=6, padx=15, pady=15)

        self.y_end_text = "Y_end:"
        y_end_label = ttk.Label(master=self.root, text=self.y_end_text, 
                                justify="left", bootstyle="default")
        y_end_label.grid(column=5, row=7, padx=0, pady=0)

        y_end_entry = tk.Entry(master=self.root, bg="white", 
                               textvariable=self.label_y_end)
        y_end_entry.grid(column=6, row=7, padx=15, pady=15)

        self.class_text = "Class:"
        class_label = ttk.Label(master=self.root, text=self.class_text, 
                                justify="left", bootstyle="default")
        class_label.grid(column=5, row=8, padx=0, pady=0)

        class_entry = tk.Entry(master=self.root, bg="white", 
                               textvariable=self.label_class)
        class_entry.grid(column=6, row=8, padx=15, pady=15)

    def open_image(self):
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

        filename = filedialog.askopenfilename(filetypes=[("JPG Files", ".jpg"),
                                                         ("PNG Files", ".png"),
                                                         ("All Files", ".*")])
        
        # Resetting self.cv_image if an image has been opened before,
        # to prevent drawing labels on top of label drawn image
        if self.cv_image is not None:
            self.cv_image = None

        if filename != "":
            self.chosen_img_label.config(text=f"Chosen Image: {filename}")
            self.chosen_image_name = os.path.split(filename)[1]
    
            self.cv_image = cv2.imread(filename)

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
            Messagebox.show_error(message="No image chosen to predict.",
                                  title="No Image to Predict")
            return
        
        if self.chosen_model.get() == "yolov8s":
            self.predict_with_yolo()
        elif self.chosen_model.get() == "faster_rcnn":
            self.predict_with_faster_rcnn()
        elif self.chosen_model.get() == "retina_net":
            self.predict_with_retina_net()
        else:
            print("Unknown choice!!!")
    
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
        
        if self.pred_complete_label.cget("text") != "":
            self.pred_complete_label.config(text="")

        model = YOLO(self.yolo_weights_dir)

        results = model.predict(self.cv_image, stream=self.stream, 
                                imgsz=self.imgsz, show=self.show, 
                                line_width=self.line_width)
        
        self.pred_complete_label.config(text="Prediction complete.")

        # To prevent multiple predictions from piling on a list
        if self.pred_labels != []:
            self.pred_labels = []
        
        for result in results:
            boxes_cls = result.boxes.cls.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

            counter = 1
            self.pred_labels.append(self.chosen_image_name)
            pred_labels = []
            
            for (box_cls, box_xyxy) in zip(boxes_cls, boxes_xyxy):
                pred_label_info = []
                pred_label_info.append(counter)
                pred_label_info.append(int(box_cls))
                pred_label_info.append(int(box_xyxy[0]))
                pred_label_info.append(int(box_xyxy[1]))
                pred_label_info.append(int(box_xyxy[2]))
                pred_label_info.append(int(box_xyxy[3]))

                pred_labels.append(pred_label_info)
                self.pred_label_ids.append(counter)

                counter += 1

            self.pred_labels.append(pred_labels)
            print(self.pred_labels)

            if self.original_pred_labels_len is None:
                self.original_pred_labels_len = len(self.pred_labels)

    def predict_with_faster_rcnn(self, num_classes=3):
        """
        Loads weight in self.faster_rcnn_weights_dir to 
        fasterrcnn_resnet50_fpn_v2, performs prediction on PIL image
        derived from self.cv_image 

        Parameters
        ----------
        num_classes : int, default=3
            Number of output classes (including background) for 
            FastRCNNPredictor

        Returns
        ----------
        None
        """

        if self.pred_complete_label.cget("text") != "":
            self.pred_complete_label.config(text="")

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

        # To prevent multiple predictions from piling on a list
        if self.pred_labels != []:
            self.pred_labels = []

        for result in results:
            boxes = result["boxes"].tolist()
            scores = result["scores"]
            classes = result["labels"].tolist()

            counter = 1
            self.pred_labels.append(self.chosen_image_name)
            pred_labels = []
            
            for (cls, score, box) in zip(classes, scores, boxes):
                if score > 0.5:
                    pred_label_info = []

                    # Makes sperm class 0, non-sperm 1
                    cls = 0 if cls == 1 else 1

                    pred_label_info.append(counter)
                    pred_label_info.append(int(cls))
                    pred_label_info.append(int(box[0]))
                    pred_label_info.append(int(box[1]))
                    pred_label_info.append(int(box[2]))
                    pred_label_info.append(int(box[3]))
                    
                    pred_labels.append(pred_label_info)
                    self.pred_label_ids.append(counter)
    
                    counter += 1

            self.pred_labels.append(pred_labels)
            print(self.pred_labels)

            if self.original_pred_labels_len is None:
                self.original_pred_labels_len = len(self.pred_labels)

    def predict_with_retina_net(self, num_classes=3):
        """
        Loads weight in self.retinanet_weights_dir to 
        retinanet_resnet50_fpn_v2, performs prediction on PIL image
        derived from self.cv_image 

        Parameters
        ----------
        num_classes : int, default=3
            Number of output classes (including background) for 
            RetinaNetClassificationHead

        Returns
        ----------
        None
        """

        if self.pred_complete_label.cget("text") != "":
            self.pred_complete_label.config(text="")

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

        # To prevent multiple predictions from piling on a list
        if self.pred_labels != []:
            self.pred_labels = []

        for result in results:
            boxes = result["boxes"].tolist()
            scores = result["scores"]
            classes = result["labels"].tolist()

            counter = 1
            self.pred_labels.append(self.chosen_image_name)
            pred_labels = []
            
            for (cls, score, box) in zip(classes, scores, boxes):
                if score > 0.5:
                    pred_label_info = []

                    # Makes sperm class 0, non-sperm 1
                    cls = 0 if cls == 1 else 1

                    pred_label_info.append(counter)
                    pred_label_info.append(int(cls))
                    pred_label_info.append(int(box[0]))
                    pred_label_info.append(int(box[1]))
                    pred_label_info.append(int(box[2]))
                    pred_label_info.append(int(box[3]))
                    
                    pred_labels.append(pred_label_info)
                    self.pred_label_ids.append(counter)
    
                    counter += 1

            self.pred_labels.append(pred_labels)
            print(self.pred_labels)

            if self.original_pred_labels_len is None:
                self.original_pred_labels_len = len(self.pred_labels)

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
            Messagebox.show_error(message="No image chosen to draw labels.",
                                  title="No Image to Draw")
            return
        
        copy_image = self.cv_image.copy()

        if self.pred_labels != [] and \
              self.chosen_image_name == self.pred_labels[0]:
            for pred_label in self.pred_labels[1]:
                counter = pred_label[0]
                pred_class = pred_label[1]
                x_start = pred_label[2]
                y_start = pred_label[3]
                x_end = pred_label[4]
                y_end = pred_label[5]
    
                if pred_class == 0:
                    cv2.rectangle(copy_image, (x_start, y_start), (x_end, y_end),
                              color=(255, 0, 0), thickness=2)
                else:
                    cv2.rectangle(copy_image, (x_start, y_start), (x_end, y_end),
                              color=(0, 255, 0), thickness=2)
                    
                cv2.putText(copy_image, str(counter), (x_end, y_start),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            color=(0, 255, 255), fontScale=1.5, thickness=3)
                
        rgb_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (768, 432))

        # Written self.pil_image to prevent Garbage Collector from
        # deleting function scope image
        # Read "Displaying Image In Tkinter Python" article in C# Corner
        # website for more info, link in "Resources Used" at the top
        self.pil_image = ImageTk.PhotoImage(Image.fromarray(rgb_image))

        self.canvas1.create_image(10, 0, anchor=NW, image=self.pil_image)

        copy_image = None

    def delete_labels(self):
        """
        Deletes label with label_id chosen by user from self.pred_labels

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.cv_image is None:
            Messagebox.show_error(message="No image chosen to delete labels.",
                                  title="No Image to Delete Labels")
            self.label_num_del.set("")
            return
        
        if self.pred_labels == []:
            Messagebox.show_error(message="No prediction performed to delete \
                                          labels from.",
                                  title="No Label to Delete")
            self.label_num_del.set("")
            return
        
        label_id = self.check_user_input(label_name=self.id_to_del_text, 
                                         input_string_var=self.label_num_del, 
                                         is_adding=False)
        
        if label_id == -1:
            return
        
        # Compared runtime speeds of iterating through list with 
        # a while loop and a for loop with self.pred_labels from image
        # dataset-testing\vid_4\images\Vid0180_8.jpg and haven't seen
        # a noticable difference, both loops gives 0.0 seconds

        for i, pred_label in zip(range(len(self.pred_labels)), self.pred_labels):
            if label_id == pred_label[0]:
                self.pred_labels.pop(i)
                self.pred_label_ids.pop(i)

        Messagebox.show_info(message=f"The Label Deleted is Label-{label_id}",
                             title="Deleted Label ID")
            
        self.label_num_del.set("")

        # To update image automatically
        self.draw_labels()

    def check_user_input(self, label_name: str, input_string_var, 
                         is_adding: bool = False, is_x_or_y: bool = True,
                         is_class: bool = False) -> int:
        """
        Checks if user input is a positive integer, then checks if user
        input is within image size or if user input is in 
        self.pred_label_ids according to is_adding variable

        Parameters
        ----------
        label_name : str
            Text of Label object located next to Entry object in GUI
        input_string_var : StringVar
            StringVar of Entry object, StringVar carries user input
        is_adding : bool, Default=False
            Shows whether user is trying to add a label or delete one.
            True for adding a label, False for deleting a label.
        is_x_or_y : bool, Default=True
            For adding a label, shows whether the input shows an x-axis
            or y-axis value. True for x-axis value, False for y-axis 
            value.
        is_class : bool, Default=False
            For adding a label, shows whether the input shows a class
            or not. True for class value, False for a not-class value.

        Returns
        ----------
        int
            Content of user input if it passes the tests, -1 if user 
            input fails a test
        """

        label_name = label_name.strip(":")

        # label_name = "Label ID to Delete:" if not is_adding
        if not is_adding:
            label_name = label_name.removesuffix(" to Delete")

        input_str = input_string_var.get()

        if not input_str.isdigit():
            Messagebox.show_error(message=label_name + " must be a positive number",
                                  title=label_name + " Isn't a Number")
            input_string_var.set("")
            return -1
        
        if is_adding and is_x_or_y and int(input_str) > self.cv_image.shape[1]:
            Messagebox.show_error(message=label_name + f" can't be greater than \
                                          image size {self.cv_image.shape[1]}",
                                  title=label_name + " Is Too Big")
            input_string_var.set("")
            return -1
        
        if is_adding and not is_x_or_y and \
            int(input_str) > self.cv_image.shape[0]:
            Messagebox.show_error(message=label_name + f" can't be greater than \
                                          image size {self.cv_image.shape[0]}",
                                  title=label_name + " Is Too Big")
            input_string_var.set("")
            return -1
        
        if is_adding and is_class and (int(input_str) < 0 or int(input_str) > 1):
            Messagebox.show_error(message=label_name + " can either be 0 (Sperm)\
                                   or 1 (Non-sperm)",
                                  title=label_name + " Is an Invalid Class")
            input_string_var.set("")
            return -1
        
        if not is_adding and int(input_str) not in self.pred_label_ids:
            Messagebox.show_error(message=label_name + " is an invalid ID",
                                  title=label_name + " is Invalid ID")
            self.label_num_del.set("")
            return -1
        
        return int(input_str)

    def add_labels(self):
        """
        Adds label with locations and type chosen by user from 
        self.pred_labels

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.cv_image is None:
            Messagebox.show_error(message="No image chosen to add labels.",
                                  title="No Image to Add Labels")
            self.label_num_del.set("")
            return
        
        if self.pred_labels == []:
            Messagebox.show_error(message="No prediction performed to add \
                                          labels to.",
                                  title="No Label to Add")
            self.label_num_del.set("")
            return
        
        label_x_start = self.check_user_input(label_name=self.x_start_text,
                                              input_string_var=self.label_x_start,
                                              is_adding=True, is_x_or_y=True)
        
        if label_x_start == -1:
            return
        
        label_y_start = self.check_user_input(label_name=self.y_start_text,
                                              input_string_var=self.label_y_start,
                                              is_adding=True, is_x_or_y=False)
        
        if label_y_start == -1:
            return
        
        label_x_end = self.check_user_input(label_name=self.x_end_text,
                                              input_string_var=self.label_x_end,
                                              is_adding=True, is_x_or_y=True)
        
        if label_x_end == -1:
            return
        
        label_y_end = self.check_user_input(label_name=self.y_end_text,
                                              input_string_var=self.label_y_end,
                                              is_adding=True, is_x_or_y=False)
        
        if label_y_end == -1:
            return
        
        label_class = self.check_user_input(label_name=self.class_text,
                                              input_string_var=self.label_class,
                                              is_adding=True, is_class=True)
        
        if label_class == -1:
            return

        labels_count = len(self.pred_labels)
        label_id = self.pred_labels[labels_count-1][0] + 1

        pred_label_info = []

        pred_label_info.append(label_id)
        pred_label_info.append(label_class)
        pred_label_info.append(label_x_start)
        pred_label_info.append(label_y_start)
        pred_label_info.append(label_x_end)
        pred_label_info.append(label_y_end)
        
        self.pred_labels.append(pred_label_info)
        self.pred_label_ids.append(label_id)
        pred_label_info = []

        Messagebox.show_info(message=f"The Label Added is Label-{label_id}",
                             title="Added Label ID")
        
        self.label_x_start.set("")
        self.label_y_start.set("")
        self.label_x_end.set("")
        self.label_y_end.set("")
        self.label_class.set("")

        # To update image automatically
        self.draw_labels()

    def choose_save_method(self):
        """
        Chooses a file format to save labels according to user's choice 
        of the option menu

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        if self.cv_image is None:
            Messagebox.show_error(message="No image chosen to save labels of.",
                                  title="No Image to Save Labels")
            return
        
        if self.pred_labels == []:
            Messagebox.show_error(message="No prediction performed to save \
                                          labels of.",
                                  title="No Label to Save")
            return

        save_type = self.chosen_save_type.get()
        
        if save_type == "txt":
            self.save_txt()
        elif save_type == "txt_yolo":
            self.save_txt(is_yolo_format=True)
        elif save_type == "json":
            self.save_json()
        else:
            print("Unknown format!!!")
    
    def save_txt(self, is_yolo_format: bool = False):
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
        is_yolo_format: boot, default=False
            Shows whether TXT file should be saved in YOLO format or not

        Returns
        ----------
        None
        """

        labels_str = ""

        for pred_label in self.pred_labels:
            pred_class = pred_label[1]
            x_start = pred_label[2]
            y_start = pred_label[3]
            x_end = pred_label[4]
            y_end = pred_label[5]

            if is_yolo_format:
                image_width = self.imgsz[0]
                image_height = self.imgsz[1]

                box_width = x_end - x_start
                box_height = y_end - y_start

                x_middle = x_start + (box_width // 2)
                y_middle = y_start + (box_height // 2)

                x_middle = x_middle / image_width
                y_middle = y_middle / image_height

                box_width = box_width / image_width
                box_height = box_height / image_height

                labels_str += str(pred_class)
                labels_str += " "
                labels_str += str(x_middle)
                labels_str += " "
                labels_str += str(y_middle)
                labels_str += " "
                labels_str += str(box_width)
                labels_str += " "
                labels_str += str(box_height)
                labels_str += "\n"
            else:
                labels_str += str(pred_class)
                labels_str += " "
                labels_str += str(x_start)
                labels_str += " "
                labels_str += str(y_start)
                labels_str += " "
                labels_str += str(x_end)
                labels_str += " "
                labels_str += str(y_end)
                labels_str += "\n"

        file_types = [("Text files", "*.txt"), ("All files", "*.*")]

        txt_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=file_types)
        
        if txt_path:
            try:
                with open(txt_path, "w") as file:
                    file.write(labels_str)
            except Exception as e:
                Messagebox.show_error(message=f"Error: {e}",
                                      title="Error While Saving TXT")
    
    def save_json(self):
        """
        Saves labels in JSON format

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        
        labels_list = []

        for pred_label in self.pred_labels:
            labels_dict = {}

            counter = pred_label[0]
            pred_class = pred_label[1]
            x_start = pred_label[2]
            y_start = pred_label[3]
            x_end = pred_label[4]
            y_end = pred_label[5]

            pred_class_str = "Sperm" if pred_class == 0 else "Non-Sperm"

            labels_dict = {"id": counter, "class": pred_class_str, 
                           "x_start": x_start, "y_start": y_start,
                           "x_end": x_end, "y_end": y_end}
            labels_list.append(labels_dict)

        file_types = [("JSON files", "*.json"), ("All files", "*.*")]

        json_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=file_types)
        
        if json_path:
            try:
                with open(json_path, "w") as file:
                    json.dump(labels_list, file, indent=4)
            except Exception as e:
                Messagebox.show_error(message=f"Error: {e}",
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
        
        choice = Messagebox.show_question(title="Exiting...",
                                          message="Are you sure you want to exit?",
                                          buttons=["Yes:primary", "No:secondary"])
        
        if choice == "Yes":
            self.root.destroy()
    

if __name__ == "__main__":
    nn_gui = NeuralNetworkGUI()

    nn_gui.root.mainloop()