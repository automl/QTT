import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, StackDataset, random_split

from qtt.data.utils import IterLoader, dict_collate, dict_tensor_to_device

from .models import MLP
from .predictor import Predictor
from .utils import MetricLogger

logger = logging.getLogger(__name__)


class CostPredictor(Predictor, torch.nn.Module):
    def __init__(
        self,
        in_dim: int | list[int],
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_metafeat_dim: int | None = None,
        out_metafeat_dim: int = 4,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            for key in kwargs:
                logger.info(
                    f"CostPredictor.__init__() got an unexpected keyword argument: {key}"
                )
        if isinstance(in_dim, int):
            in_dim = [in_dim]
        self.in_dim = in_dim

        # build config encoder
        encoder = nn.ModuleList()
        for dim in in_dim:
            encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
        self.config_encoder = encoder
        enc_dims = len(in_dim) * enc_out_dim

        if in_metafeat_dim is not None:
            self.meta_encoder = MLP(in_metafeat_dim, out_metafeat_dim)
            enc_dims += out_metafeat_dim
        else:
            self.meta_encoder = None

        self.head = MLP(enc_dims, 1, enc_nlayers, enc_hidden_dim, act_fn=nn.GELU)

    @torch.no_grad()
    def predict(self, config, metafeat=None, **kwarg):
        self.eval()
        return self(config, metafeat).detach().cpu().numpy().squeeze()

    def forward(self, config, metafeat=None, **kwargs):
        # encode config
        x = []
        start = 0
        for i, dim in enumerate(self.in_dim):
            end = start + dim
            output = self.config_encoder[i](config[:, start:end])  # type: ignore
            x.append(output)
            start = end
        x = torch.cat(x, dim=1)

        if self.meta_encoder is not None:
            out = self.meta_encoder(metafeat)
            x = torch.cat([x, out], dim=1)

        x = self.head(x)
        x = nn.functional.relu(x)  # cost is always positive
        return x

    def fit(
        self,
        dataset: Dataset,
        bs: int = 256,
        lr: float = 1e-3,
        train_steps: int = 10_000,
        val_freq: int = 100,
        use_scheduler: bool = False,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/cost_predictor",
        ckpt_name: str = "cost.pth",
        log_freq: int = 10,
    ):
        return self._fit(
            dataset,
            bs,
            lr,
            train_steps,
            val_freq,
            use_scheduler,
            test_size,
            seed,
            cache_dir,
            ckpt_name,
            log_freq,
        )

    def update(
        self,
        data: Dataset | dict,
        bs: int = 64,
        lr: float = 1e-4,
        train_steps: int = 100,
        val_freq: int = 10,
        use_scheduler: bool = False,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/cost_predictor",
        ckpt_name: str = "cost.pth",
        log_freq: int = 10,
    ):
        if isinstance(data, dict):
            dataset = StackDataset(**data)
        return self._fit(
            dataset,
            bs,
            lr,
            train_steps,
            val_freq,
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
        bs: int = 128,
        lr: float = 1e-3,
        train_steps: int = 10_000,
        val_freq: int = 10,
        use_scheduler: bool = False,
        test_size: float = 0.2,
        seed: int | None = None,
        cache_dir: str = "~/.cache/qtt/cost_predictor",
        ckpt_name: str = "cost.pth",
        log_freq: int = 10,
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
        )

        # ============ preparing optimizer ... ============
        optimizer = torch.optim.AdamW(self.parameters(), lr)
        scheduler = None
        if use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, train_steps, eta_min=1e-6)

        # ============ training ... ============
        torch.save(self.state_dict(), save_path)
        min_error = self.__validate(dev, val_loader)
        print(f"Initial validation error: {min_error:.3f}")
        self.train()
        metric_log = MetricLogger(delimiter="  ")
        for it, batch in enumerate(
            metric_log.log_every(train_loader, log_freq, "TRAIN"), 1
        ):
            batch = dict_tensor_to_device(batch, dev)
            config, metafeat, cost = batch["config"], batch["metafeat"], batch["cost"]

            # forward + loss
            output = self(config, metafeat)
            loss = torch.nn.functional.mse_loss(output, cost)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() if scheduler is not None else None            

            # logging
            metric_log.update(loss=loss.item())

            # validation
            if not it % val_freq:
                self.eval()
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
        val_loss = []
        for batch in val_loader:
            batch = dict_tensor_to_device(batch, dev)
            config, metafeat, cost = batch["config"], batch["metafeat"], batch["cost"]
            output = self(config, metafeat)
            loss = torch.nn.functional.l1_loss(output, cost)
            val_loss.append(loss.item())
        val_error = sum(val_loss) / len(val_loss)
        return val_error

    def save(self, path: str | Path, name: str = "cost_predictor.pth"):
        path = Path(path)
        torch.save(self.state_dict(), path / name)

    def load(self, path: str | Path, name: str = "cost_predictor.pth"):
        path = Path(path)
        ckp = torch.load(path / name)
        self.load_state_dict(ckp)
        return self
