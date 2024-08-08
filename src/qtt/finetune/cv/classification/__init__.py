import os
from pathlib import Path

from torchvision.datasets import ImageFolder

from .finetune_wrapper import finetune_script

__all__ = ["finetune_script"]


def extract_task_info_metafeat(
    root: str | Path, train_split: str = "train", val_split: str = "val"
):
    root = Path(root)
    assert root.exists(), f"dataset-path: {root} does not exist."

    num_samples = 0
    num_classes = 0
    num_features = 128  # fixed for now
    num_channels = 3

    # trainset
    train_path = os.path.join(root, train_split)
    if os.path.exists(train_path):
        trainset = ImageFolder(train_path)
        num_samples += len(trainset)
        num_channels = 3 if trainset[0][0].mode == "RGB" else 1
        num_classes = len(trainset.classes)

    # valset
    val_path = os.path.join(root, val_split)
    if os.path.exists(val_path):
        valset = ImageFolder(val_path)
        num_samples += len(valset)

    metafeat = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "num_features": num_features,
        "num_channels": num_channels,
    }

    task_info = {
        "data-path": root,
        "train-split": train_split,
        "val-split": val_split,
        "num-classes": num_classes,
    }

    return task_info, metafeat
