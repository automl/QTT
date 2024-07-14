import os
import time
from collections import deque
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from qtt.data.dataset import MetaDataset
from qtt.optimizers.surrogates.dyhpo import DyHPO


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


def process_curve(curve):
    """Randomly sample a budget and set the scores after the budget to 0."""
    BS, N = curve.shape
    # len of curves
    budget = (curve != 0).sum(dim=1)
    # sample random budget
    rnd_bdgt = torch.randint_like(budget, low=1, high=N)
    budget = torch.min(budget, rnd_bdgt)
    # target is the score at the budget
    target = curve[torch.arange(curve.size(0)), budget - 1]
    # set the scores after the budget to 0
    rows = torch.arange(N).expand(BS, N).to(curve.device)
    indices = budget.view(-1, 1).expand(BS, N) - 2
    mask = rows > indices
    curve[mask] = 0
    return curve, budget, target


def metatrain_dyhpo(
    dyhpo: DyHPO,
    dataset: Dataset,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_iter: int = 10000,
    val_freq: int = 100,
    val_iter: int = 10,
    test_size: float = 0.2,
    use_scheduler: bool = True,
    device: str = "auto",
    cache_dir="~/.cache/qtt/metatrain",
    ckpt_name="dyhpo.pth",
    seed: Optional[int] = None,
    log_freq: int = 50,
):
    # ============ setup cache dir and device ... ============
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, ckpt_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dyhpo.to(dev)

    # ============ preparing data ... ============
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    trainset, valset = random_split(
        dataset=dataset,
        lengths=[1 - test_size, test_size],
        generator=generator,
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
    )

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(dyhpo.parameters(), lr)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)

    # ============ training ... ============
    min_loss = float("inf")
    loss_history = deque(maxlen=20)
    dyhpo.train()
    start_time = time.time()
    for it, batch in enumerate(IterationWrapper(train_loader, train_iter)):
        it += 1

        batch = [item.to(dev) for item in batch]
        config, score, metafeat, _ = batch

        curve, budget, target = process_curve(score)

        train_x = {
            "config": config,
            "metafeat": metafeat,
            "budget": budget,
            "curve": curve,
        }
        loss = dyhpo.train_step(train_x, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_history.append(loss.item())
        if not it % log_freq:
            _loss = sum(loss_history) / len(loss_history)
            elapsed = time.time() - start_time
            eta = elapsed / it * (train_iter - it)
            print(
                f"TRAIN  [{it:}/{train_iter}]:",
                f"loss: {_loss:.3f}",
                f"lenghtscale: {dyhpo.gp_model.covar_module.base_kernel.lengthscale.item():.3f}",
                f"noise: {dyhpo.gp_model.likelihood.noise.item():.3f}",  # type: ignore
                f"eta: {eta:.2f}s",
                sep="  ",
            )

        if not it % val_freq:
            dyhpo.eval()
            val_loss = []
            for batch in IterationWrapper(val_loader, val_iter):
                batch = [item.to(dev) for item in batch]
                config, score, metafeat, _ = batch
                curve, budget, target = process_curve(score)
                batch = {
                    "config": config,
                    "metafeat": metafeat,
                    "budget": budget,
                    "curve": curve,
                }
                pred = dyhpo.predict(batch)
                mean = pred.mean
                loss = torch.nn.functional.l1_loss(mean, target)
                val_loss.append(loss.item())

            val_error = sum(val_loss) / len(val_loss)
            if val_error < min_loss:
                min_loss = val_error
                torch.save(dyhpo.state_dict(), save_path)
                print(f"VAL  [{it}]: Checkpoint saved with val-error: {val_error:.3f}")
            else:
                print(f"VAL  [{it}]: val-error: {val_error:.3f}")
            dyhpo.train()

    # Load the model with the best validation error
    print(f"Loading the model with the best validation error: {min_loss:.3f}")
    dyhpo.load_state_dict(torch.load(os.path.join(cache_dir, ckpt_name)))

    return dyhpo


def metatrain_cost_estimator(
    model: torch.nn.Module,
    dataset: MetaDataset,
    batch_size: int = 128,
    lr: float = 1e-3,
    grad_clip: float = 0.0,
    train_iter: int = 10000,
    val_freq: int = 100,
    test_size: float = 0.3,
    use_scheduler: bool = True,
    device="auto",
    cache_dir="~/.cache/qtt/meta",
    ckpt_name="ckpt.pth",
    seed: Optional[int] = None,
    log_freq: int = 50,
):
    # ============ setup cache dir and device ... ============
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, ckpt_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    # ============ preparing data ... ============
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    trainset, valset = random_split(
        dataset=dataset,
        lengths=[1 - test_size, test_size],
        generator=generator,
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
    )

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)

    # ============ training ... ============
    min_loss = float("inf")
    loss_history = deque(maxlen=20)
    loader = iter(train_loader)
    model.train()
    start_time = time.time()
    for it in range(1, train_iter + 1):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(train_loader)
            batch = next(loader)

        batch = [item.to(device) for item in batch]
        config, _, metafeat, cost = batch
        output = model(config, metafeat)
        loss = torch.nn.functional.mse_loss(output, cost)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_history.append(loss.item())
        if it % log_freq == 0:
            _loss = sum(loss_history) / len(loss_history)
            elapsed = time.time() - start_time
            eta = elapsed / it * (train_iter - it)
            print(
                f"TRAIN [{it}/{train_iter}]:",
                f"loss: {_loss:.3f}",
                f"eta: {eta:.2f}s",
                sep="  ",
            )

        if not it % val_freq:
            model.eval()
            val_loss = []
            for batch in val_loader:
                batch = [item.to(device) for item in batch]
                config, _, metafeat, cost = batch
                target = cost

                output = model(config, metafeat)
                loss = torch.nn.functional.l1_loss(output, target)
                val_loss.append(loss.item())

            val_error = sum(val_loss) / len(val_loss)
            if val_error < min_loss:
                min_loss = val_error
                torch.save(model.state_dict(), save_path)
                print(f"VAL  [{it}]: Checkpoint saved with val-error: {val_error:.3f}")
            else:
                print(f"VAL  [{it}]: val-error: {val_error:.3f}")

            model.train()

    # Load the model with the best validation error
    print(f"Loading the model with the best validation error: {min_loss:.3f}")
    model.load_state_dict(torch.load(save_path))

    return model
