import os
import torch

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2


def getCOCO(datadir: str, image_size: int, batchsize = 64):
    """Prepares a dataloader for the 2017 validation set of COCO."""

    root = os.path.join(datadir, 'images', 'val2017')
    annFile = os.path.join(datadir, 'annotations', 'instances_val2017.json')

    transforms = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    coco_dataset = CocoDetection(root=root, annFile=annFile, transforms=transforms)
    coco_dataset = datasets.wrap_dataset_for_transforms_v2(coco_dataset, target_keys=['boxes', 'labels'])

    coco_dataloader = DataLoader(coco_dataset, batch_size=batchsize, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

    return coco_dataloader