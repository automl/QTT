import copy
import json
import logging
import os
import time
from collections import deque

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from .models import MLP
from .predictor import Predictor

logger = logging.getLogger(__name__)


class CostPredictor(Predictor):
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

    @classmethod
    def from_pretrained(cls, root: str) -> "CostPredictor":
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(**config)
        model_path = os.path.join(root, "estimator.pth")
        model._load_meta_checkpoint(model_path)
        return model

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

    def _load_meta_checkpoint(self, path: str):
        self.meta_checkpoint = path
        self.meta_trained = True
        state = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {path} with msg: {msg}")

    def update(self, data: dict, hp: dict | None = None):
        if hp is None:
            hp = {}
        steps = hp.get("steps", 50)
        lr = hp.get("lr", 1e-3)

        state_dict = copy.deepcopy(self.state_dict())

        target = data.pop("cost")

        pred = self(**data)
        pre_acc = torch.nn.functional.l1_loss(pred, target)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr)

        for _ in range(steps):
            optimizer.zero_grad()
            output = self(**data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        self.eval()
        pred = self(**data)
        final_acc = torch.nn.functional.l1_loss(pred, target)

        if final_acc > pre_acc:
            self.load_state_dict(state_dict)
            print("Update failed, reverting to previous state.")

        print(f"Accuracy before: {pre_acc:.4f}, after: {final_acc:.4f}")

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
        cache_dir: str = "~/.cache/qtt/cost_predictor",
        ckpt_name: str = "cost.pth",
        log_freq: int = 10,
    ):
        # ============ setup cache dir and device ... ============
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, ckpt_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr)
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

            batch = move_dict_to_device(batch, device)
            config = batch["config"]
            metafeat = batch["metafeat"]
            target = batch["cost"]

            output = self(config, metafeat)
            loss = torch.nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_history.append(loss.item())
            if it % log_freq == 0:
                _loss = sum(loss_history) / len(loss_history)
                elapsed = time.time() - start_time
                eta = elapsed / it * (n_iter - it)
                print(
                    f"TRAIN [{it}/{n_iter}]:",
                    f"loss: {_loss:.3f}",
                    f"eta: {eta:.2f}s",
                    sep="  ",
                )

            if not it % val_freq:
                self.eval()
                val_loss = []
                for batch in IterationWrapper(val_loader, val_iter):
                    batch = move_dict_to_device(batch, device)
                    config = batch["config"]
                    metafeat = batch["metafeat"]
                    target = batch["cost"]

                    output = self(config, metafeat)
                    loss = torch.nn.functional.l1_loss(output, target)
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
        self.load_state_dict(torch.load(save_path))

        return self


def dict_collate_fn(batch):
    """Collate function for DataLoader."""
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0].keys()}


def move_dict_to_device(data, device):
    """Move dictionary to device."""
    return {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}


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
