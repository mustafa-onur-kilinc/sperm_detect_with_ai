"""
Script by Özgün Zeki BOZKURT.
"""

import torch

from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from functools import partial


def get_model(model_name, backbone_name, weights_path, train=False, 
              num_classes=3):
    if model_name == "retinanet":
        model = retinanet_resnet50_fpn_v2(weights=None)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )

    if model_name == "faster-rcnn":
        if backbone_name == "resnet50":
            model = fasterrcnn_resnet50_fpn_v2(weights=None)

        if backbone_name == "mobilenet":
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 
                                                          num_classes)

    if train == False:
        model.load_state_dict(torch.load(weights_path))
    return model


if __name__ == "__main__":
    model = get_model(
        "retinanet", "resnet50", 
        r"sperm-detection\runs\train\run_7\RetinaNet_resnet50_epoch_18.pth"
    )
    print(model)
