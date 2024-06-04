"""
Script by Özgün Zeki BOZKURT. 

Made very minor changes by Mustafa Onur KILINÇ to obey 
PEP8 maximum line length. Didn't test the script after making these 
changes but considering they are minor changes like defining string 
variables, turning ternary operators to if-else blocks and adding 
newlines before parameters of functions, it should be fine.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from get_model import get_model
from dataset import CustomDataset


import numpy as np
import random
from datetime import datetime
import os
from tqdm import tqdm
from utils.check_dir import get_next_run_directory
from utils.gpu_cooldown import cool_down_if_needed


# To align batch input sizes
def custom_collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    model_name,
    backbone_name,
    dataset_path,
    save_dir=r"sperm_detection/runs/train",
    hyperparams=None,
    resize=(1080, 1920),
    save_period=2,
):
    set_seed(42)
    if hyperparams == None:
        hyperparams = {
            "epoch": 20,
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "batch_size": 4,
            "step_size": 5,
            "gamma": 0.1,
        }
        print("No hyperparameters specified running with default hyperparameters: ")

    print("Resize scale: ", resize)
    os.makedirs(save_dir, exist_ok=True)
    runs_dir = get_next_run_directory(save_dir)
    os.makedirs(runs_dir, exist_ok=True)

    # Specific dataset directory structure
    images_train_path = os.path.join(dataset_path, "train", "images")
    labels_train_path = os.path.join(dataset_path, "train", 
                                     "labels-corner-coordinates")

    images_val_path = os.path.join(dataset_path, "test", "images")
    labels_val_path = os.path.join(dataset_path, "test", 
                                   "labels-corner-coordinates")

    # Perform resizing, convert images to tensors, normalize RGB values into [0,1]
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    # Specify training and validation dataloaders
    dataset_train = CustomDataset(
        images_train_path, labels_train_path, transforms=transform
    )
    dataset_val = CustomDataset(
        images_val_path, labels_val_path, transforms=transform
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=hyperparams.get("batch_size"),
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=hyperparams.get("batch_size"),
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    ##########################

    # Broke ternary operator used here to obey PEP8 max line length
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else: 
        torch.device("cpu")

    model = get_model(model_name, backbone_name, weights_path=None, train=True)
    model.to(device)

    print("Using device: ", device)
    print(f"Model: {model._get_name()} with backbone: {backbone_name}")
    print(hyperparams)

    params = [p for p in model.parameters() if p.requires_grad]

    # Define SGD optimizer with hyperparameters
    optimizer = torch.optim.SGD(
        params,
        lr=hyperparams["learning_rate"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
    )

    scheduler = StepLR(
        optimizer,
        step_size=hyperparams.get("step_size"),
        gamma=hyperparams.get("gamma"),
    )

    num_epochs = hyperparams.get("epoch")

    # Initialize lists to record losses
    training_losses = []
    validation_losses = []
    start_time = datetime.now()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets, _ in tqdm(
            data_loader_train, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            # Transfer images and targets to the device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            cool_down_if_needed()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], \
                Training Loss: {running_loss/len(data_loader_train):.4f}"
        )

        avg_training_loss = running_loss / len(data_loader_train)
        training_losses.append(avg_training_loss)

        validation_loss = 0.0
        for images, targets, _ in tqdm(
            data_loader_val,
            f"Epoch {epoch+1}/{num_epochs} (val)",
        ):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                validation_loss += losses.item()

        # Calculate and print validation loss
        validation_loss /= len(data_loader_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], \
              Validation Loss: {validation_loss:.4f}")
        validation_losses.append(validation_loss)

        scheduler.step()

        end_time = datetime.now()

        training_duration = end_time - start_time

        with open(os.path.join(runs_dir, "training_info.txt"), "w") as f:
            f.write(f"{model._get_name()}_{backbone_name}\n\n")
            for param, value in hyperparams.items():
                f.write(f"{param}: {value}\n")

            f.write(f"\nDataset used: {dataset_path}")
            f.write(f"\nInput resize: {resize}")

            # Defined formatted_start_time to obey PEP8 max line length
            formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(
                f"\n\nTraining Start Time: {formatted_start_time}\n"
            )

            # Defined formatted_end_time to obey PEP8 max line length
            formatted_end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Training End Time: {formatted_end_time}\n")
            f.write(f"Training Duration: {training_duration}\n")

            f.write("\nTraining losses: ")
            for loss in training_losses:
                f.write(f"{loss}, ")
            f.write("\nValidation losses: ")
            for loss in validation_losses:
                f.write(f"{loss}, ")

            f.write("\n")

        # Save the model
        if (epoch + 1) % save_period == 0:
            # Defined weight_name to obey PEP8 maximum line length
            weight_name = f"{model._get_name()}_{backbone_name}_epoch_{epoch+1}"
            weight_name += ".pth"

            torch.save(model.state_dict(), f"{runs_dir}/{weight_name}")


if __name__ == "__main__":
    model_name = "faster-rcnn"
    backbone_name = "resnet50"
    dataset_path = r"dataset/dataset-split1"

    hyperparams = {
        "epoch": 30,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "batch_size": 2,
        "step_size": 5,
        "gamma": 0.1,
    }

    train(
        model_name=model_name,
        backbone_name=backbone_name,
        dataset_path=dataset_path,
        hyperparams=hyperparams,
        resize=(1080, 1920),
        save_period=1,
    )
