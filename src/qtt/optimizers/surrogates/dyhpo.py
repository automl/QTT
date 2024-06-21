import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import inspect

import gpytorch
import torch

from .encoder import FeatureEncoder

logger = logging.getLogger("dyhpo")


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


class DyHPO(torch.nn.Module):
    refine_steps = 50
    refine_lr = 1e-3

    def __init__(
        self,
        in_features: int | List[int],
        **kwargs,
    ):
        super().__init__()
        enc_kwargs = dict()
        for key, value in kwargs.items():
            if key in inspect.signature(FeatureEncoder.__init__).parameters.keys():
                enc_kwargs[key] = value
            else:
                logger.warning(f"Unknown parameter {key} for DyHPO.")

        self.encoder = FeatureEncoder(in_features, **enc_kwargs)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = GPRegressionModel(
            train_x=None,
            train_y=None,
            likelihood=self.likelihood,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp_model,
        )

    @torch.no_grad()
    def predict(self, data: Dict[str, torch.Tensor]):
        proj = self.encoder(**data)
        pred = self.likelihood(self.gp_model(proj))
        return pred

    def train_step(self, train_x: Dict[str, torch.Tensor], train_y: torch.Tensor):
        train_x = self.encoder(**train_x)
        self.gp_model.set_train_data(train_x, train_y, False)
        output = self.gp_model(train_x)
        loss = -self.mll(output, train_y)  # type: ignore
        return loss

    def fit_pipeline(self, data: Dict[str, torch.Tensor]):
        self.train()

        # create optimizer
        optimizer = torch.optim.Adam(self.parameters(), self.refine_lr)

        target = data.pop("target")

        for i in range(self.refine_steps):
            optimizer.zero_grad()
            feat = self.encoder(**data)
            self.gp_model.set_train_data(feat, target, False)
            output = self.gp_model(feat)
            loss = -self.mll(output, target)  # type: ignore
            loss.backward()
            optimizer.step()

            # print(
            #     "Iter %2d/%2d - Loss: %1.3f lengthscale: %1.3f noise: %1.3f"
            #     % (
            #         i + 1,
            #         self.train_steps,
            #         loss.item(),
            #         self.model.covar_module.base_kernel.lengthscale.item(),
            #         self.model.likelihood.noise.item(),  # type: ignore
            #     )
            # )

        self.eval()

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        train_y = train_data.pop("target")
        with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
            # train_x = self.encoder(**train_data)
            # self.gp_model.set_train_data(train_x, train_y, False)

            test_x = self.encoder(**test_data)
            pred = self.likelihood(self.gp_model(test_x))

        mean = pred.mean.reshape(-1)
        std = pred.stddev.reshape(-1)
        return mean, std

    @classmethod
    def from_pretrained(cls, root: str) -> "DyHPO":
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
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(**config)
        model_path = os.path.join(root, "dyhpo.pth")
        model.load_meta_checkpoint(model_path)
        return model

    # def _get_state(self):
    #     state = deepcopy(self.state_dict())
    #     return state

    # def save_checkpoint(self, path: str = ".", with_config: bool = True):
    #     """
    #     Save the state to a checkpoint file.

    #     Args
    #     ----
    #     checkpoint : str
    #         The path to the checkpoint file.
    #     """
    #     state = self.state_dict()
    #     save_path = os.path.join(path, "surrogate.pth")
    #     torch.save(state, save_path)
    #     logger.info(f"Saved model to {save_path}")
    #     if with_config:
    #         config = {
    #             "extractor_cfg": self.extractor_cfg,
    #             "predictor_cfg": self.predictor_cfg,
    #         }
    #         config_path = os.path.join(path, "config.json")
    #         with open(config_path, "w") as f:
    #             json.dump(config, f, indent=2)
    #         logger.info(f"Saved config to {config_path}")

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

