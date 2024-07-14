import copy
import inspect
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

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
    meta_trained = False
    train_steps = 1000
    train_lr = 1e-3
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
                logger.warning(f"Unknown parameter '{key}' for DyHPO.")

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
        state_dict = copy.deepcopy(self.state_dict())

        target = data.pop("target")
        pred = self.predict(data)
        loss_pre = (pred.mean - target).abs().mean()

        self.train()

        # create optimizer
        lr = self.refine_lr if self.meta_trained else self.train_lr
        optimizer = torch.optim.Adam(self.parameters(), lr)

        steps = self.refine_steps if self.meta_trained else self.train_steps
        for i in range(steps):
            optimizer.zero_grad()
            feat = self.encoder(**data)
            self.gp_model.set_train_data(feat, target, False)
            output = self.gp_model(feat)
            loss = -self.mll(output, target)  # type: ignore
            loss.backward()
            optimizer.step()

            # logger.debug(
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

        pred = self.predict(data)
        loss_aft = (pred.mean - target).abs().mean()

        if loss_aft > loss_pre:
            self.load_state_dict(state_dict)
            logger.debug("no improvement, reverting to previous state.")
        logger.debug(f"Loss before: {loss_pre:.4f}, after: {loss_aft:.4f}")

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        train_data.pop("target")
        with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
            # train_x = self.encoder(**train_data)
            # self.gp_model.set_train_data(train_x, train_y, False)

            test_x = self.encoder(**test_data)
            pred = self.likelihood(self.gp_model(test_x))

        mean = pred.mean.reshape(-1)
        std = pred.stddev.reshape(-1)
        return mean, std

    @classmethod
    def from_pretrained(cls, root: str, ckp_name="dyhpo.pth") -> "DyHPO":
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(**config)
        ckp_path = os.path.join(root, ckp_name)
        model._load_meta_checkpoint(ckp_path)
        return model

    def _load_meta_checkpoint(self, path: str):
        self.meta_checkpoint = path
        self.meta_trained = True
        state = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {path} with msg: {msg}")
