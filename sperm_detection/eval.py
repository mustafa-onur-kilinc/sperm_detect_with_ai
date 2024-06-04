"""
Script to evaluate model performance on test dataset.
Outputs result on runs/eval directory.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
from get_model import get_model

from torchvision import transforms
from dataset import CustomDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
from utils.check_dir import get_next_run_directory

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_eval(ground_truths_path, predictions_path, save_dir):
    coco_gt = COCO(ground_truths_path)  # Load ground truth
    coco_dt = coco_gt.loadRes(predictions_path)  # Load detections
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    output_f_path = os.path.join(save_dir, "evaluation_results.txt")

    original_stdout = sys.stdout
    with open(output_f_path, "w") as f:
        sys.stdout = f
        coco_eval.summarize()

    sys.stdout = original_stdout

    print("COCO evaluation results saved to:", output_f_path)


def custom_collate_fn(batch):
    return tuple(zip(*batch))


def eval_model(
    model_name,
    backbone_name,
    weights_path,
    val_dir,
    batch_size=1,
    resize=(1080, 1920),
    n_c=3,
):
    # Load model

    # Broke ternary operator used here into if-else block to obey PEP8
    # maximum line length 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        torch.device("cpu")

    model = get_model(model_name, backbone_name, weights_path)
    model.eval()
    model.to(device)

    img_dir = os.path.join(val_dir, "images")
    labels_dir = os.path.join(val_dir, "labels-corner-coordinates")

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_val = CustomDataset(img_dir, labels_dir, transforms=transform)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    # COCO format label info placeholder
    categories_info = []
    for i in range(1, n_c + 1):
        categories_info.append({"id": i, str(i): str(i)})

    predictions = []
    ground_truths = []
    images_info = []
    with torch.no_grad():
        for images, targets, image_names in tqdm(
            data_loader_val, desc="Collecting predictions"
        ):

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i, output in enumerate(outputs):
                output = {k: v.to("cpu") for k, v in output.items()}
                image_id = targets[i]["image_id"].item()  # Extract image_id

                # COCO image metadata.
                images_info.append(
                    {"id": image_id, image_names[i]: f"image_{image_id}.jpg"}
                )

                # Collect predictions
                pred_boxes = output["boxes"].numpy().tolist()
                pred_scores = output["scores"].numpy().tolist()
                pred_labels = output["labels"].numpy().tolist()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    # Convert corner coordinate bbox format to 
                    # COCO format (x, y, W, H)
                    box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    predictions.append(
                        {
                            "image_id": targets[i]["image_id"].item(),
                            "category_id": int(label),  # category = label
                            "bbox": box_coco,
                            "score": score,
                        }
                    )

                # Collect ground truth annotations
                gt_boxes = targets[i]["boxes"].cpu().numpy().tolist()
                gt_labels = targets[i]["labels"].cpu().numpy().tolist()
                for box, label in zip(gt_boxes, gt_labels):
                    box_coco = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    ground_truths.append(
                        {
                            "image_id": targets[i]["image_id"].item(),
                            "category_id": int(label),
                            "bbox": box_coco,
                            "iscrowd": 0,
                            "area": (box[2] - box[0])
                            * (box[3] - box[1]),  # Optional, for completeness
                            "id": len(ground_truths) + 1,  # Assign a unique ID
                        }
                    )

    # Construct the complete ground_truths dictionary in COCO format
    ground_truths_coco = {
        "images": images_info,
        "annotations": ground_truths,
        "categories": categories_info,
    }

    save_dir = get_next_run_directory(r"sperm_detection\runs\eval")
    os.makedirs(save_dir, exist_ok=True)
    weights_name = os.path.basename(weights)
    weights_name = os.path.splitext(weights_name)[0]

    preds_save = os.path.join(save_dir, f"predictions_{weights_name}.json")

    ground_truths_save = os.path.join(save_dir, "ground_truths.json")

    model_info_save = os.path.join(save_dir, "model_info.txt")
    
    # Save predictions 
    # (no change needed, as predictions format is correct for pycocotools)
    with open(preds_save, "w") as f:
        json.dump(predictions, f)

    # Save ground_truths in the COCO-required structure
    with open(ground_truths_save, "w") as f:
        json.dump(ground_truths_coco, f)

    with open(model_info_save, "w") as f:
        f.write(f"{model._get_name()}_resnet50\n")
        f.write(f"Weights path: {weights}\n")
        f.write(f"Dataset path: {val_dir}")

    print("Prediction results saved on: ", save_dir)
    coco_eval(ground_truths_save, preds_save, save_dir)


if __name__ == "__main__":
    # Sample Usage
    script_dir = os.path.dirname(__file__)
    main_dir = os.path.split(script_dir)[0]

    model_name = "retinanet"
    backbone_name = "resnet50"
    weights = os.path.join(main_dir, "weights", "retina_net",
                           "RetinaNet_resnet50_epoch_18.pth")

    val_dir = os.path.join(main_dir, "dataset-split2", "test")
    eval_model(model_name, backbone_name, weights, val_dir, resize=(1080, 1920))

    for i in range(1, 12):
        val_dir = os.path.join(main_dir, "bitirme-1", "telefon-dataset-testing", 
                               f"vid_{i}")
        eval_model(model_name, backbone_name, weights, val_dir, 
                   resize=(1080, 1920))
