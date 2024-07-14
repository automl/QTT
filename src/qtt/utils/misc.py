import logging
import os
import pandas as pd
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def extract_image_dataset_metadata(
    path: str, train_split: str = "train", val_split: str = "val"
):
    assert os.path.exists(path), f"dataset-path: {path} does not exist."
    n_samples = 0
    n_classes = 0
    n_features = 128  # fixed for now
    n_channels = 3

    # trainset
    train_path = os.path.join(path, train_split)
    if os.path.exists(train_path):
        trainset = ImageFolder(train_path)
        n_samples += len(trainset)
        n_channels = 3 if trainset[0][0].mode == "RGB" else 1
        n_classes = len(trainset.classes)

    # valset
    val_path = os.path.join(path, val_split)
    if os.path.exists(val_path):
        valset = ImageFolder(val_path)
        n_samples += len(valset)

    df = pd.DataFrame(
        {
            "n_samples": [n_samples],
            "n_classes": [n_classes],
            "n_features": [n_features],
            "n_channels": [n_channels],
        }
    )

    return df
