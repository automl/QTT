import copy
import logging
import os
from typing import Tuple

import gpytorch
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from .encoder import FeatureEncoder
from .predictor import Predictor

logger = logging.getLogger(__name__)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor | None,
        train_y: torch.Tensor | None,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)  # type: ignore


class DyHPO(Predictor):
    def __init__(
        self,
        in_dim: int | list[int],
        out_dim: int = 32,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_curve_dim: int = 50,
        out_curve_dim: int = 16,
        curve_channels: int = 1,
        in_metafeat_dim: int | None = None,
        out_metafeat_dim: int = 16,
        *,
        train_steps=1000,
        train_lr=1e-3,
        refine_steps=50,
        refine_lr=1e-3,
    ):
        super().__init__()

        self.encoder = FeatureEncoder(
            in_dim,
            out_dim,
            enc_hidden_dim,
            enc_out_dim,
            enc_nlayers,
            in_curve_dim,
            out_curve_dim,
            curve_channels,
            in_metafeat_dim,
            out_metafeat_dim,
        )
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

        self.train_steps = train_steps
        self.train_lr = train_lr
        self.refine_steps = refine_steps
        self.refine_lr = refine_lr

    def forward(self, data: dict[str, torch.Tensor]):
        return self.gp_model(self.encoder(**data))

    @torch.no_grad()
    def predict(self, data: dict[str, torch.Tensor]):
        return self.likelihood(self(data))

    def train_step(self, X: dict[str, torch.Tensor], y: torch.Tensor):
        enc = self.encoder(**X)
        self.gp_model.set_train_data(enc, y, False)
        output = self.gp_model(enc)
        loss = -self.mll(output, y)  # type: ignore
        return loss

    def fit_pipeline(self, data: dict[str, torch.Tensor]):
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

    def _predict_pipeline(
        self,
        train_data: dict[str, torch.Tensor],
        test_data: dict[str, torch.Tensor],
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

    def fit(
        self,
        dataset: Dataset,
        bs: int = 32,
        lr: float = 1e-3,
        n_iter: int = 10_000,
        val_freq: int = 100,
        val_iter: int = 100,
        use_scheduler: bool = False,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/dyhpo",
        ckpt_name: str = "dyhpo.pth",
        log_freq: int = 10,
    ):
        import time
        from collections import deque

        # ============ setup cache dir and device ... ============
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, ckpt_name)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        # ============ preparing data ... ============
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        trainset, valset = random_split(
            dataset=dataset,
            lengths=[1 - test_size, test_size],
            generator=generator,
        )
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            collate_fn=dict_collate_fn,
        )
        val_loader = DataLoader(
            dataset=valset,
            batch_size=bs,
            collate_fn=dict_collate_fn,
        )

        # ============ preparing optimizer ... ============
        optimizer = torch.optim.Adam(self.parameters(), lr)
        scheduler = None
        if use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, n_iter, eta_min=1e-7)

        # ============ training ... ============
        min_loss = float("inf")
        loss_history = deque(maxlen=20)
        self.train()
        start_time = time.time()
        for it, batch in enumerate(IterationWrapper(train_loader, n_iter)):
            it += 1

            batch = dict_to_device(batch, dev)
            config, score, metafeat = batch["config"], batch["score"], batch["metafeat"]

            curve, fidelity, target = generate_target(score)

            train_x = {
                "config": config,
                "fidelity": fidelity,
                "curve": curve,
                "metafeat": metafeat,
            }
            loss = self.train_step(train_x, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_history.append(loss.item())
            if not it % log_freq:
                _loss = sum(loss_history) / len(loss_history)
                elapsed = time.time() - start_time
                eta = elapsed / it * (n_iter - it)
                print(
                    f"TRAIN  [{it:}/{n_iter}]:",
                    f"loss: {_loss:.3f}",
                    f"lenghtscale: {self.gp_model.covar_module.base_kernel.lengthscale.item():.3f}",
                    f"noise: {self.gp_model.likelihood.noise.item():.3f}",  # type: ignore
                    f"eta: {eta:.2f}s",
                    sep="  ",
                )

            # validation
            if not it % val_freq:
                self.eval()
                val_loss = []
                for batch in IterationWrapper(val_loader, val_iter):
                    batch = dict_to_device(batch, dev)
                    config, score, metafeat = (
                        batch["config"],
                        batch["score"],
                        batch["metafeat"],
                    )
                    curve, fidelity, target = generate_target(score)
                    batch = {
                        "config": config,
                        "metafeat": metafeat,
                        "fidelity": fidelity,
                        "curve": curve,
                    }
                    pred = self.predict(batch)
                    loss = torch.nn.functional.l1_loss(pred.mean, target)
                    val_loss.append(loss.item())

                val_error = sum(val_loss) / len(val_loss)
                if val_error < min_loss:
                    min_loss = val_error
                    torch.save(self.state_dict(), save_path)
                    print(
                        f"VAL  [{it}]: Checkpoint saved with val-error: {val_error:.3f}"
                    )
                else:
                    print(f"VAL  [{it}]: val-error: {val_error:.3f}")
                self.train()

        # Load the model with the best validation error
        print(f"Loading the model with the best validation error: {min_loss:.3f}")
        self.load_state_dict(torch.load(os.path.join(cache_dir, ckpt_name)))

        return self


def dict_collate_fn(batch):
    """Collate function for DataLoader."""
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0].keys()}


def dict_to_device(data, device):
    """Move dictionary to device."""
    return {k: v.to(device) for k, v in data.items()}


def generate_target(curve):
    """Generate target from curve."""
    BS, N = curve.shape
    # len of curves
    fidelity = (curve != 0).sum(dim=1)
    # sample random fidelity
    rnd_bdgt = torch.randint_like(fidelity, low=1, high=N)
    fidelity = torch.min(fidelity, rnd_bdgt)
    # target is the score at the fidelity
    target = curve[torch.arange(curve.size(0)), fidelity - 1]
    # set the scores after the fidelity to 0
    rows = torch.arange(N).expand(BS, N).to(curve.device)
    indices = fidelity.view(-1, 1).expand(BS, N) - 2
    mask = rows > indices
    curve[mask] = 0
    return curve, fidelity, target


class IterationWrapper:
    def __init__(self, loader, n):
        self.loader = loader
        self.iterator = iter(loader)
        self.n = n
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.step += 1
        if self.step > self.n:
            raise StopIteration
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
