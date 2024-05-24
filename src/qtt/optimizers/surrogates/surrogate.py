from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Surrogate(nn.Module):
    """
    A base class for surrogate models used in optimization algorithms.
    """

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "Surrogate":
        """
        Load a pretrained surrogate model from the given path.

        Args
        ----
        path : str
            The path to the pretrained model.

        Returns
        -------
        Surrogate
            The pretrained surrogate model.
        """
        raise NotImplementedError

    def predict_pipeline(
        self, train_data: Dict[str, torch.Tensor], test_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform the prediction pipeline using the surrogate model.

        Args
        ----
        train_data : Dict[str, torch.Tensor])
            The training data used for model fitting.
        test_data : Dict[str, torch.Tensor]
            The test data used for prediction.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            The mean, deviation and cost predictions of the given data.
        """
        raise NotImplementedError

    def train_pipeline(
        self, data: Dict[str, torch.Tensor], restart: bool = False
    ) -> None:
        """
        Refine the surrogate model using the given training data.

        Args
        ----
        data : Dict[str, torch.Tensor]
            The training data used for model fitting.
        """
        raise NotImplementedError
