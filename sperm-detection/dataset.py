from torchvision.transforms.functional import to_tensor, resize
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms if transforms is not None else ToTensor()
        self.imgs = list(sorted(os.listdir(images_dir)))
        self.labels = list(sorted(os.listdir(labels_dir)))

    def __getitem__(self, idx):
        # Load images and labels
        img_path = os.path.join(self.images_dir, self.imgs[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        img = Image.open(img_path).convert("RGB")

        original_size = img.size

        # Apply transformations
        if self.transforms is not None:
            img = self.transforms(img)

        new_size = (img.shape[2], img.shape[1])

        boxes, labels = self._read_labels(label_path)

        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]

        # print(scale_x, scale_y)

        boxes = [
            [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]
            for xmin, ymin, xmax, ymax in boxes
        ]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img_name = self.imgs[idx]

        return img, target, img_name

    def _read_labels(self, label_path):
        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f:
                class_label, xmin, ymin, xmax, ymax = line.strip().split()
                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(int(class_label))

        return boxes, labels

    def __len__(self):
        return len(self.imgs)
