import json
import logging
import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import gpytorch
import torch

from qtt.optimizers.surrogates.models import CostPredictor, FeatureExtractor
from qtt.optimizers.surrogates.surrogate import Surrogate

logger = logging.getLogger("dyhpo")


class DyHPO(Surrogate):
    """
    The DyHPO DeepGP model. This version of DyHPO also includes a Cost Predictor.
    """

    lr = 1e-4
    train_steps: int = 1000
    refine_steps: int = 50
    meta_trained: bool = False
    meta_checkpoint: Optional[str] = None

    def __init__(
        self,
        extractor_cfg: dict,
        predictor_cfg: dict,
        **kwargs,
    ):
        super().__init__()
        self.extractor_cfg = extractor_cfg
        self.predictor_cfg = predictor_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor: FeatureExtractor
        self.cost_predictor: CostPredictor
        self.gp: GPRegressionModel
        self.gll: gpytorch.likelihoods.GaussianLikelihood
        self.mll: gpytorch.mlls.ExactMarginalLogLikelihood
        self._reinit_()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter {key} for DyHPO.")

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "DyHPO":
        """
        Load a pretrained DyHPO model from a checkpoint.

        Args
        ----
        path : str
            The path to the checkpoint file.

        Returns
        -------
        DyHPO
            The loaded DyHPO model.
        """
        root = name_or_path
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(config["extractor_cfg"], config["predictor_cfg"])
        model_path = os.path.join(root, "surrogate.pth")
        model.load_meta_checkpoint(model_path)
        return model

    def _reinit_(self):
        """
        Restart the surrogate model from scratch.
        """
        if self.meta_checkpoint is not None:
            self.load_meta_checkpoint(self.meta_checkpoint)
        else:
            self.feature_extractor = FeatureExtractor(**self.extractor_cfg)
            self.cost_predictor = CostPredictor(**self.predictor_cfg)
            self.gll = gpytorch.likelihoods.GaussianLikelihood()
            self.gp = GPRegressionModel(None, None, self.gll)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gll, self.gp)
        self.to(self.device)

    def save_checkpoint(self, path: str = ".", with_config: bool = True):
        """
        Save the state to a checkpoint file.

        Args
        ----
        checkpoint : str
            The path to the checkpoint file.
        """
        state = self.state_dict()
        save_path = os.path.join(path, "surrogate.pth")
        torch.save(state, save_path)
        logger.info(f"Saved model to {save_path}")
        if with_config:
            config = {
                "extractor_cfg": self.extractor_cfg,
                "predictor_cfg": self.predictor_cfg,
            }
            config_path = os.path.join(path, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved config to {config_path}")


    def load_meta_checkpoint(self, path: str):
        """
        Load the state from a checkpoint.

        Args
        ----
        checkpoint : str
            The path to the checkpoint file.
        """
        self.meta_checkpoint = path
        self.meta_trained = True
        state = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {path} with msg: {msg}")

    def forward(
        self,
        config: torch.Tensor,
        budget: torch.Tensor,
        curve: torch.Tensor,
        target: torch.Tensor,
        metafeat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        proj = self.feature_extractor(config, budget, curve, metafeat)

        self.gp.set_train_data(proj, target, False)
        output = self.gp(proj)

        loss = -self.mll(output, target)  # type: ignore
        return loss

    def train_pipeline(self, data: Dict[str, torch.Tensor], restart: bool = False):
        """
        Trains the surrogate with the provided data.

        Args
        ----
        data : Dict[str, torch.Tensor]
            A dictionary containing the input data for training.
            It should contain the following keys:
                - config: The hyperparameters configurations.
                - budget: The budget values.
                - curve: The learning curves.
                - target: The target values.
                - metafeat: The metafeatures.
        """
        self.train()

        config = data["config"]
        if config.size(0) == 1:  # skip training if only one point is provided
            return

        optimizer = torch.optim.Adam(self.parameters(), self.lr)

        initial_state = self._get_state()
        training_errored = False

        for key, item in data.items():
            data[key] = item.to(self.device)

        if restart:
            self._reinit_()

        for _ in range(self.refine_steps):
            try:
                loss = self.forward(**data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                logger.warn(f"The following error happened while training: {e}")
                self.restart = True
                training_errored = True
                break

        if training_errored:
            self.load_state_dict(initial_state)

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the target values for the test data.

        Args
        ----
        train_data : Dict[str, torch.Tensor]
            A dictionary containing the input data for training.
            It should contain the following
                - config: The hyperparameters configurations.
                - target: The target values.
                - budget: The budget values.
                - curve: The learning curves.
                - metafeat: The metafeatures.
        test_data : Dict[str, torch.Tensor]
            A dictionary containing the input data for testing.
            It should contain the following
                - config: The hyperparameters configurations.
                - budget: The budget values.
                - curve: The learning curves.
                - target: The target values.
                - metafeat: The metafeatures.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the predicted means, the predicted standard deviations
            and the predicted costs.
        """
        self.eval()

        for key, item in train_data.items():
            if item is not None:
                train_data[key] = item.to(self.device)
        for key, item in test_data.items():
            if item is not None:
                test_data[key] = item.to(self.device)

        target = train_data.pop("target")
        test_data.pop("target", None)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_feat = self.feature_extractor(**train_data)
            self.gp.set_train_data(train_feat, target, False)

            test_feat = self.feature_extractor(**test_data)
            pred = self.gll(self.gp(test_feat))

            cost = self.cost_predictor(**test_data)

        mean = pred.mean.reshape(-1)
        std = pred.stddev.reshape(-1)

        return mean, std, cost

    def _get_state(self):
        state = deepcopy(self.state_dict())
        return state


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: Optional[torch.Tensor],
        train_y: Optional[torch.Tensor],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)  # type: ignore
