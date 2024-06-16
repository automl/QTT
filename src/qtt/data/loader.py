import random
from typing import Optional

from sklearn.model_selection import train_test_split
import torch

from .dataset import MTLBMDataSet


class MetaDataLoader:
    """
    Data loader for the meta dataset, that generates batches for training or evaluation.

    Args:
        dataset (MetaSet): The meta dataset to load data from.
        batch_size (int): The batch size.
        seed (Optional[int]): The random seed for splitting the dataset into train and validation sets.
    """

    def __init__(
        self,
        dataset: MTLBMDataSet,
        batch_size: int,
        test_size: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        datasets = self.dataset.get_datasets()
        self.train_split, self.val_split = train_test_split(
            datasets, test_size=test_size, random_state=seed
        )

    def get_batch(
        self, mode: str = "train", metric: str = "perf"
    ) -> dict[str, torch.Tensor]:
        """
        Get a batch of data from the dataset.

        Args:
            mode (str): The mode of operation. Can be either "train" or "val".
            metric (str): The evaluation metric to use.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the batch of data.

        Raises:
            AssertionError: If an unknown mode is provided.
        """
        assert mode in ["train", "val"], f"Unknown mode: {mode}"

        if mode == "train":
            dname = random.choice(self.train_split)
            return self.dataset.get_batch(self.batch_size, metric, dname)
        else:
            dname = random.choice(self.val_split)
            return self.dataset.get_batch(self.batch_size, metric, dname)
