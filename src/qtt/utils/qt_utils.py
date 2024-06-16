import logging
import os
from dataclasses import dataclass
from enum import Enum

from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


class QTaskStatus(Enum):
    SUCCESS = 1
    ERROR = 2


@dataclass
class QTunerResult:
    idx: int
    score: float
    time: float
    status: QTaskStatus
    info: str = ""


def get_dataset_metafeatures(
    path: str, train_split: str = "train", val_split: str = "val"
):
    """
    Get metafeatures of the dataset.

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    dict
        meta-features of the dataset
    """
    logger.info("metafeatures not given, infer from dataset")
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

    meta_features = [n_samples, n_classes, n_features, n_channels]
    return meta_features
