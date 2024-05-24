import logging
import os
import sys
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
    perf: float
    time: float
    status: QTaskStatus
    info: str = ""


def get_dataset_metafeatures(path: str):
    """
    Get metafeatures of the dataset.

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    dict
        Metafeatures of the dataset
    """
    logger.info("metafeatures not given, infer from dataset")
    if not path.endswith("train"):
        path = os.path.join(path, "train")
    assert os.path.exists(path), f"Path {path} does not exist."
    try:
        dataset = ImageFolder(path)
        num_samples = len(dataset)
        num_channels = 3 if dataset[0][0].mode == "RGB" else 1
        num_classes = len(dataset.classes)
        image_size = dataset[0][0].size[1]
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    meta_features = {
        "n_samples": num_samples,
        "n_classes": num_classes,
        "n_features": image_size,
        "n_channels": num_channels,
    }
    return meta_features
