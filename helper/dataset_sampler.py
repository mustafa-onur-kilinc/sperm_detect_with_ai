"""
This Python script is used to sample images and label TXT files from a 
source directory and copy them to a destination directory. The sampling 
rate, source directory and destination directory is defined in the 
YAML file.

Note: This script doesn't have the ability to cycle through different
folders in dataset directory and can only work with a single 
patient_name/(Ekran or Telefon)/(labels_src_dir or images_src_dir) 
directory. The user has to manually change patient_name, 
is_src_name_telefon and images_src_name variables to perform sampling
in a dataset with more than one patient or video.

In order to work properly, this script needs a folder structure like 
this:

- main_directory
    - config_folder
        - yaml_name
    - helpers_folder
        - dataset_sampler.py
    - dataset_dir
        - patient_name
            - Ekran
                - labels_src_dir
                    - 'video_name'_1.txt
                    - 'video_name'_2.txt
                    - 'video_name'_3.txt
                    - (...)
                - images_src_dir
                    - 'video_name'_1.jpg
                    - 'video_name'_2.jpg
                    - 'video_name'_3.jpg
                    - (...)
            - Telefon
                - labels_src_dir
                    - 'video_name'_1.txt
                    - 'video_name'_2.txt
                    - 'video_name'_3.txt
                    - (...)
                - images_src_dir
                    - 'video_name'_1.jpg
                    - 'video_name'_2.jpg
                    - 'video_name'_3.jpg
                    - (...)
    - dest_dir (optional)
        - train_dir (optional)
            - train_images_dir (optional)
            - train_labels_dir (optional)
        - test_dir (optional)
            - test_images_dir (optional)
            - test_labels_dir (optional)
        - valid_dir (optional)
            - valid_images_dir (optional)
            - valid_labels_dir (optional)

Destination directory and its subdirectories will be created by the code 
if they don't exist.

Resources used to write this script:
James Mertz n.d., "Documenting Python Code: A Complete Guide", 
Real Python, accessed 28 March 2024, 
<https://realpython.com/documenting-python-code/>

numpydoc maintainers n.d., "Style guide", accessed 28 March 2024, 
<https://numpydoc.readthedocs.io/en/latest/format.html#style-guide>

Python 3.12.2 Documentation 2024, "8. Errors and Exceptions", 
Python Software Foundation, accessed 28 March 2024, 
<https://docs.python.org/3/tutorial/errors.html#handling-exceptions>

Brady Huang 2021, 
"How to count the number of files in a directory using Python", 
Stack Exchange Inc., accessed 28 March 2024, 
<https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python#comment116024313_2632251>

MaxU - stand with Ukraine 2016, 
"How to split data into 3 sets (train, validation and test)?",
Stack Exchange Inc., accessed 28 March 2024, 
<https://stackoverflow.com/a/38251213>

this be Shiva 2019, "Convert pandas dataframe to NumPy array", 
Stack Exchange Inc., accessed 28 March 2024, 
<https://stackoverflow.com/a/54508052>

pandas.Series.to_numpy n.d., pandas, accessed 28 March 2024,
<https://pandas.pydata.org/docs/reference/api/pandas.Series.to_numpy.html>

sailees14032000 2021, 
"Python – Copy contents of one file to another file", GeeksforGeeks, 
accessed 28 March 2024, 
<https://www.geeksforgeeks.org/python-copy-contents-of-one-file-to-another-file/>

Tormod Landet 2018, "yaml.load does not support encodings different from 
current system encoding, cannot you add it?", GitHub Inc., 
accessed 18 May 2024, 
<https://github.com/yaml/pyyaml/issues/123#issuecomment-395431735>
"""

import os
import cv2
import yaml
import numpy as np
import pandas as pd

class DatasetSampler():
    """
    This class has the ability to perform tasks listed below:
    - Getting paths to image and label directories in dataset 
      directories
    - Getting paths to destination directories
    - Making directories if destination directory or some of its 
      subdirectories don't exist
    - Randomizing images and labels, performing train, test and
      validation splits on them according to rates defined by user
    - Copying images and labels from dataset directories to train, test,
      validation directories in destination directory 
    """
    
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        main_dir = os.path.split(script_dir)[0]

        config_dirname = "config"
        yaml_name = "dataset_sampler.yaml"
        yaml_dir = os.path.join(config_dirname, yaml_name)

        with open(yaml_dir, encoding="utf8") as yaml_file:
            args_dict = yaml.safe_load(yaml_file)

        dataset_name = args_dict["dataset_name"]

        # List to append 'Ekran'/'Telefon' to patient name 
        src_name = args_dict["src_name"]

        # Two options: Telefon (True) or Ekran (False)
        is_src_name_telefon = args_dict["is_src_name_telefon"]  

        # Using if-else block to make choosing Telefon/Ekran less prone
        # to error and easier
        if is_src_name_telefon:
            src_name.append("Telefon")
        else:
            src_name.append("Ekran")

        # destination directory is relative to script
        dest_name = args_dict["dest_name"] 

        # Only this gets "self." because image and label names are 
        # "images_src_name" + "_number" 
        self.images_src_name = args_dict["images_src_name"]
        labels_src_name = self.images_src_name + "_labels"

        train_dir_name = args_dict["train_dir_name"]
        valid_dir_name = args_dict["valid_dir_name"]
        test_dir_name = args_dict["test_dir_name"]
        images_dst_name = args_dict["images_dst_name"]
        labels_dst_name = args_dict["labels_dst_name"]

        # If False, value of validSplitRate will be added to 
        # trainSplitRate
        self.isTrainValidSplit = args_dict["isTrainValidSplit"]  
        
        # This variable is n in 1/n sampling rate
        self.sampling_rate = args_dict["sampling_rate"]
        
        # validSplitRate will be calculated according to all of dataset, 
        # instead of training split. Eg. in order to have a %80-%20-%20 
        # training-validation-test split, you need to assign 0.64, 0.16 
        # and 0.2 values to trainSplitRate, validSplitRate and 
        # testSplitRate respectively.
        self.trainSplitRate = args_dict["trainSplitRate"]
        self.validSplitRate = args_dict["validSplitRate"]
        self.testSplitRate = args_dict["testSplitRate"]

        dataset_dir = self.getDirectory(main_dir, dataset_name)

        # This line assumes src_name is a list with at least two variables
        src_dir = self.getDirectory(dataset_dir, src_name[0], src_name[1]) 

        self.images_src_dir = self.getDirectory(src_dir, self.images_src_name)
        self.labels_src_dir = self.getDirectory(src_dir, labels_src_name)

        try:
            dest_dir = self.getDirectory(main_dir, dest_name)
        except OSError:
            dest_dir = self.makeDirectory(main_dir, dest_name)

        try:
            train_dir = self.getDirectory(dest_dir, train_dir_name)
        except OSError:
            train_dir = self.makeDirectory(dest_dir, train_dir_name)

        try:
            test_dir = self.getDirectory(dest_dir, test_dir_name)
        except OSError:
            test_dir = self.makeDirectory(dest_dir, test_dir_name)
        
        if self.isTrainValidSplit:
            try:
                valid_dir = self.getDirectory(dest_dir, valid_dir_name)
            except OSError:
                valid_dir = self.makeDirectory(dest_dir, valid_dir_name)

        try:
            self.train_images_dir = self.getDirectory(train_dir, 
                                                      images_dst_name)
        except OSError:
            self.train_images_dir = self.makeDirectory(train_dir, 
                                                       images_dst_name)

        try:
            self.test_images_dir = self.getDirectory(test_dir, images_dst_name)
        except OSError:
            self.test_images_dir = self.makeDirectory(test_dir, images_dst_name)
        
        if self.isTrainValidSplit:
            try:
                self.valid_images_dir = self.getDirectory(valid_dir, 
                                                          images_dst_name)
            except OSError:
                self.valid_images_dir = self.makeDirectory(valid_dir, 
                                                           images_dst_name)

        try:
            self.train_labels_dir = self.getDirectory(train_dir, 
                                                      labels_dst_name)
        except OSError:
            self.train_labels_dir = self.makeDirectory(train_dir, 
                                                       labels_dst_name)

        try:
            self.test_labels_dir = self.getDirectory(test_dir, 
                                                     labels_dst_name)
        except OSError:
            self.test_labels_dir = self.makeDirectory(test_dir, 
                                                      labels_dst_name)
        
        if self.isTrainValidSplit:
            try:
                self.valid_labels_dir = self.getDirectory(valid_dir, 
                                                          labels_dst_name)
            except OSError:
                self.valid_labels_dir = self.makeDirectory(valid_dir, 
                                                           labels_dst_name)

    def getDirectory(self, source_dir: str, child_dir: str, 
                     child_dir_2: str | None = None) -> str:
        """
        Combines source_dir, path2 and path3, checks if the resulting 
        directory exists and returns directory if it does.

        This function takes two or three string variables, combines them 
        to get a directory using os.path.join() function, checks if the 
        resulting directory exists using os.path.exists() function. If 
        the resulting directory doesn't exist, this function will raise 
        an OSError. If the resulting directory exists, this function 
        will return it.

        Parameters
        ----------
        source_dir : String
            Source directory to add directory names to
        child_dir : String
            Name of a child directory to add to source directory
        child_dir_2 : String, optional
            Name of a child directory of child_dir to add to source_dir 
            and child_dir. Defaults to None

        Returns
        ----------
        directory : String
            A string pointing to a directory, combination of source_dir, 
            child_dir and child_dir_2

        Raises
        ----------
        OSError
            If combination of source_dir, child_dir and child_dir_2 is 
            an invalid directory
        """

        if child_dir_2 is not None:
            directory = os.path.join(source_dir, child_dir, child_dir_2)
        else:
            directory = os.path.join(source_dir, child_dir)

        if not os.path.exists(directory):
            raise OSError(f"{directory} doesn't exist!!!")

        return directory

    def makeDirectory(self, source_dir: str, child_dir: str, 
                      child_dir_2: str | None = None):
        """
        Combines source_dir, path2 and path3, makes the resulting 
        directory and returns it.

        This function takes two or three string variables, combines them
        to get a directory using os.path.join() function, tries to make 
        the resulting directory using os.mkdir() function. If the 
        resulting directory is invalid or an already existing one, this 
        function will raise the resulting OSError. If the resulting 
        directory is a valid directory, this function will return it.

        Parameters
        ---------- 
        source_dir : String
            Source directory to add directory names to
        child_dir : String
            Name of a child directory to add to source directory
        child_dir_2 : String, optional
            Name of a child directory of child_dir to add to source_dir 
            and child_dir

        Returns
        ----------
        directory : String
            A string pointing to a directory, combination of source_dir, 
            child_dir and child_dir_2

        Raises
        ----------
        OSError
            If combination of source_dir, child_dir and child_dir_2 is 
            an invalid or existing directory
        """

        if child_dir_2 is not None:
            directory = os.path.join(source_dir, child_dir, child_dir_2)
        else:
            directory = os.path.join(source_dir, child_dir)

        try:
            os.mkdir(directory)
        except OSError as error:
            raise(error)
        
        return directory
    
    def getDatasetSplit(self):
        """
        Counts images in self.images_src_dir, samples every one in 
        self.sampling_rate, performs training-validation-test split to 
        image indices list, returns resulting index arrays. 

        This function counts from 1 to image_count + 1, appends every 
        one in self.sampling_rate to image_number_list, turns that list 
        into a Pandas DataFrame, performs training-validation-test split 
        to that DataFrame, transforms resulting three dataFrames to 
        Numpy arrays and returns those arrays.

        TODO: Add option to perform training-test split according to 
        self.isTrainValidSplit

        Parameters
        ---------- 
        None

        Returns
        ----------
        train_img_arr : NDArray
            Includes image and label numbers picked for training set
        test_img_arr : NDArray
            Includes image and label numbers picked for test set
        validate_img_arr : NDArray
            Includes image and label numbers picked for validation set

        Notes
        ----------
        Performing train-val-test split to a Pandas DataFrame is from 
        this source:
        
        MaxU - stand with Ukraine 2016, 
        "How to split data into 3 sets (train, validation and test)?",
        Stack Exchange Inc., accessed 28 March 2024, 
        <https://stackoverflow.com/a/38251213>
        """

        image_number_list = []
        image_count = len(os.listdir(self.images_src_dir))

        for counter in range(1, image_count+1):
            if counter % self.sampling_rate == 0:
                image_number_list.append(counter)

        df = pd.DataFrame(image_number_list)

        (train_set_arr, 
        validate_set_arr, 
        test_set_arr) = np.split(df.sample(frac=1, random_state=42),
                                [int(self.trainSplitRate * len(df)),
                                int((1 - self.testSplitRate) * len(df))])
                
        train_set_arr = train_set_arr.to_numpy()
        validate_set_arr = validate_set_arr.to_numpy()
        test_set_arr = test_set_arr.to_numpy()
        
        return train_set_arr, test_set_arr, validate_set_arr

    def copyDataset(self):
        """
        Gets training-validation-test indices from self.getDatasetSplit() 
        function, copies images and label files to directories defined 
        in __init__() function.
         
        This function copies images from self.images_src_dir to 
        self.train_images_dir, self.test_images_dir and if 
        self.isTrainValidSplit is True, to self.valid_images_dir 
        directories. It also copies label files from self.labels_src_dir 
        to self.train_labels_dir, self.test_labels_dir and if 
        self.isTrainValidSplit is True, to self.valid_labels_dir
        directories.

        Parameters
        ---------- 
        None

        Returns
        ----------
        None
        """

        train_set_arr, test_set_arr, validate_set_arr = self.getDatasetSplit()

        for index in range(0, len(train_set_arr)):
            image_number = train_set_arr[index][0]
            image_name = self.images_src_name + "_" + str(image_number) + ".png"
            label_name = self.images_src_name + "_" + str(image_number) + ".txt"

            # OpenCV can't open image because of image path including 
            # Turkish characters apparently
            os.chdir(self.images_src_dir)  
            sperm_image = cv2.imread(os.path.join(".", image_name))
            os.chdir(self.train_images_dir)
            cv2.imwrite(os.path.join(".", image_name), sperm_image)
            print(f"Copying {image_name} from {self.images_src_dir} to \
                  {self.train_images_dir}")
            
            label_src_dir = os.path.join(self.labels_src_dir, label_name)
            label_dst_dir = os.path.join(self.train_labels_dir, label_name)

            with (open(label_src_dir, "r") as src_label_file, 
                  open(label_dst_dir, "a") as dst_label_file):
                for line in src_label_file:
                    dst_label_file.write(line)
            print(f"Copying {label_name} from {self.labels_src_dir} to \
                  {self.train_labels_dir}")

        for index in range(0, len(test_set_arr)):
            image_number = test_set_arr[index][0]
            image_name = self.images_src_name + "_" + str(image_number) + ".png"
            label_name = self.images_src_name + "_" + str(image_number) + ".txt"

            # OpenCV can't open image because of image path including 
            # Turkish characters apparently
            os.chdir(self.images_src_dir)  
            sperm_image = cv2.imread(os.path.join(".", image_name))
            os.chdir(self.test_images_dir)
            cv2.imwrite(os.path.join(".", image_name), sperm_image)
            print(f"Copying {image_name} from {self.images_src_dir} to \
                  {self.test_images_dir}")

            label_src_dir = os.path.join(self.labels_src_dir, label_name)
            label_dst_dir = os.path.join(self.test_labels_dir, label_name)

            with (open(label_src_dir, "r") as src_label_file, 
                  open(label_dst_dir, "a") as dst_label_file):
                for line in src_label_file:
                    dst_label_file.write(line)
            print(f"Copying {label_name} from {self.labels_src_dir} to \
                  {self.test_labels_dir}")


        if self.isTrainValidSplit:
            for index in range(0, len(validate_set_arr)):
                image_number = validate_set_arr[index][0]
                image_name = self.images_src_name + "_" + str(image_number) + ".png"
                label_name = self.images_src_name + "_" + str(image_number) + ".txt"

                # OpenCV can't open image because of image path 
                # including Turkish characters apparently
                os.chdir(self.images_src_dir)  
                sperm_image = cv2.imread(os.path.join(".", image_name))
                os.chdir(self.valid_images_dir)
                cv2.imwrite(os.path.join(".", image_name), sperm_image)
                print(f"Copying {image_name} from {self.images_src_dir} to \
                      {self.valid_images_dir}")

                label_src_dir = os.path.join(self.labels_src_dir, label_name)
                label_dst_dir = os.path.join(self.valid_labels_dir, label_name)

                with (open(label_src_dir, "r") as src_label_file, 
                      open(label_dst_dir, "a") as dst_label_file):
                    for line in src_label_file:
                        dst_label_file.write(line)
                print(f"Copying {label_name} from {self.labels_src_dir} to \
                      {self.valid_labels_dir}")


if __name__ == "__main__":
    datasetSampler = DatasetSampler()

    datasetSampler.copyDataset()
    