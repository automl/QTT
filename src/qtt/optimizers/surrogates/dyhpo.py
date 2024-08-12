import logging
import os
from pathlib import Path

import gpytorch
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, StackDataset, random_split, DataLoader

from qtt.data.utils import IterLoader, dict_collate, dict_tensor_to_device

from .encoder import FeatureEncoder
from .predictor import Predictor
from .utils import MetricLogger

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


class DyHPO(Predictor, torch.nn.Module):
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

    @torch.no_grad()
    def predict(self, data: dict[str, torch.Tensor]):
        enc = self.encoder(**data)
        output = self.gp_model(enc)
        return self.likelihood(output)

    def train_step(self, X: dict[str, torch.Tensor], y: torch.Tensor):
        enc = self.encoder(**X)
        self.gp_model.set_train_data(enc, y, False)
        output = self.gp_model(enc)
        loss = -self.mll(output, y)  # type: ignore
        return loss

    def fit(
        self,
        dataset: Dataset,
        bs: int = 128,
        lr: float = 1e-3,
        train_steps: int = 10_000,
        val_freq: int = 100,
        val_steps: int = 256,
        use_scheduler: bool = True,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/dyhpo",
        ckpt_name: str = "dyhpo.pth",
        log_freq: int = 10,
    ):
        return self._fit(
            dataset,
            bs,
            lr,
            train_steps,
            val_freq,
            val_steps,
            use_scheduler,
            test_size,
            seed,
            cache_dir,
            ckpt_name,
            log_freq,
        )

    def update(
        self,
        dataset: Dataset | dict,
        bs: int = 64,
        lr: float = 1e-4,
        train_steps: int = 100,
        val_freq: int = 10,
        val_steps: int = 100,
        use_scheduler: bool = False,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/dyhpo",
        ckpt_name: str = "dyhpo.pth",
        log_freq: int = 10,
    ):
        if isinstance(dataset, dict):
            dataset = StackDataset(**dataset)

        return self._fit(
            dataset,
            bs,
            lr,
            train_steps,
            val_freq,
            val_steps,
            use_scheduler,
            test_size,
            seed,
            cache_dir,
            ckpt_name,
            log_freq,
        )

    def _fit(
        self,
        dataset: Dataset,
        bs: int,
        lr: float,
        train_steps: int,
        val_freq: int,
        val_steps: int,
        use_scheduler: bool,
        test_size: float,
        seed: int | None,
        cache_dir: str,
        ckpt_name: str,
        log_freq: int,
    ):
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
        train_loader = IterLoader(
            dataset=trainset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            collate_fn=dict_collate,
            steps=train_steps,
        )
        val_loader = DataLoader(
            dataset=valset,
            batch_size=bs,
            collate_fn=dict_collate,
            # steps=val_steps,
        )

        # ============ preparing optimizer ... ============
        optimizer = torch.optim.AdamW(self.parameters(), lr)
        scheduler = None
        if use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, train_steps, eta_min=1e-6)

        # ============ training ... ============
        # when doing update, save the model before training
        torch.save(self.state_dict(), save_path)
        min_error = self.__validate(dev, val_loader)
        print(f"Initial validation error: {min_error:.3f}")
        self.train()
        metric_log = MetricLogger(delimiter="  ")
        for it, batch in enumerate(
            metric_log.log_every(train_loader, log_freq, "TRAIN"), 1
        ):
            batch = dict_tensor_to_device(batch, dev)
            train_x, target = self.__generate_batch(batch)

            loss = self.train_step(train_x, target)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() if scheduler is not None else None

            # logging
            metric_log.update(loss=loss.item())
            metric_log.update(
                lengthscale=self.gp_model.covar_module.base_kernel.lengthscale.item()
            )
            metric_log.update(noise=self.gp_model.likelihood.noise.item())  # type: ignore

            # validation
            if not it % val_freq:
                val_error = self.__validate(dev, val_loader)
                print(f"VAL [{it}] val-error: {val_error:.3f}")
                if val_error < min_error:
                    min_error = val_error
                    torch.save(self.state_dict(), save_path)
                self.train()

        # Load the model with the best validation error
        print(f"Loading the model with the best validation error: {min_error:.3f}")
        self.load_state_dict(torch.load(save_path))
        return self

    def __validate(self, dev, val_loader):
        self.eval()
        val_loss = []
        for batch in val_loader:
            batch = dict_tensor_to_device(batch, dev)
            train_x, target = self.__generate_val_batch(batch)
            pred = self.predict(train_x)
            loss = torch.nn.functional.l1_loss(pred.mean, target)
            val_loss.append(loss.item())
        val_error = sum(val_loss) / len(val_loss)
        return val_error

    def __generate_batch(self, data: dict[str, torch.Tensor]):
        curve = data["curve"]
        BS, N = curve.shape
        # not all learning curves are fully evaluated
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

        train = {
            "config": data["config"],
            "curve": curve,
            "fidelity": fidelity / N,
            "metafeat": data.get("metafeat"),
        }
        return train, target

    def __generate_val_batch(self, data: dict[str, torch.Tensor]):
        """
        Generate validation batch for DyHPO.
        Similar as __generate_batch but without randomness.
        Validating the model on fixed NS values over the learning curve.
        """
        NS = 10  # number of samples
        config = data["config"]
        curve = data["curve"]
        metafeat = data.get("metafeat")
        BS, N = curve.shape

        # not all learning curves are fully evaluated
        max_fidelity = (curve != 0).sum(dim=1).reshape(-1, 1)
        samples = torch.arange(1, N, N // NS, device=config.device).reshape(1, -1)
        fidelity = torch.min(max_fidelity, samples)
        mask = (fidelity < max_fidelity).reshape(-1)

        config = config.repeat_interleave(NS, 0)[mask]
        curve = curve.repeat_interleave(NS, 0)[mask]
        fidelity = fidelity.reshape(-1)[mask]
        if metafeat is not None:
            metafeat = metafeat.repeat_interleave(NS, 0)
            metafeat = metafeat[mask]

        # target is the score at the fidelity
        # fidelity starts from 1, so we need to subtract 1
        target = curve[torch.arange(curve.size(0)), fidelity - 1]

        # set the curve values at the fidelity and after to 0
        row_indices = torch.arange(N, device=curve.device).unsqueeze(0)
        indices = fidelity.unsqueeze(1) - 2
        mask = (row_indices > indices.expand_as(curve))
        curve[mask] = 0

        train = {
            "config": config,
            "curve": curve,
            "fidelity": fidelity / N,
            "metafeat": metafeat,
        }
        return train, target

    def save(self, path: str | Path, name: str = "dyhpo.pth"):
        path = Path(path)
        torch.save(self.state_dict(), path / name)

    def load(self, path: str | Path, name: str = "dyhpo.pth"):
        path = Path(path)
        ckp = torch.load(path / name)
        self.load_state_dict(ckp)
        return self
