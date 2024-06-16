import os
from collections import deque
import time
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from qtt.data.dataset import MTLBMDataSet
from qtt.data.loader import MetaDataLoader
from qtt.optimizers.surrogates.dyhpo import DyHPO


def metatrain_dyhpo(
    dyhpo: DyHPO,
    metaset: MTLBMDataSet,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_iter: int = 10000,
    val_iter: int = 100,
    val_freq: int = 100,
    use_scheduler: bool = True,
    device="auto",
    cache_dir="~/.cache/qtt/metatrain",
    ckpt_name="dyhpo.pth",
    seed: Optional[int] = None,
    log_freq: int = 50,
):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, ckpt_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dyhpo.train()
    dyhpo.to(device)

    loader = MetaDataLoader(metaset, batch_size, 0.1, seed)
    optimizer = torch.optim.Adam(dyhpo.parameters(), lr)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)

    min_loss = float("inf")
    loss_history = deque(maxlen=20)
    start_time = time.time()
    for it in range(1, train_iter + 1):
        optimizer.zero_grad()
        batch = loader.get_batch(metric="perf")
        for key, item in batch.items():
            batch[key] = item.to(device)

        target = batch.pop("target")
        loss = dyhpo.train_step(batch, target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        scheduler.step() if scheduler is not None else None

        if it % log_freq == 0:
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
            val_error = 0
            for _ in range(val_iter):
                batch = loader.get_batch(mode="val")
                for key, item in batch.items():
                    batch[key] = item.to(device)
                target = batch.pop("target")
                pred = dyhpo.predict(batch)
                mean = pred.mean
                loss = torch.nn.functional.l1_loss(mean, target)
                val_error += loss.item()

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
    metaset: MTLBMDataSet,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_iter: int = 10000,
    val_iter: int = 100,
    val_freq: int = 100,
    use_scheduler: bool = True,
    device="auto",
    cache_dir="~/.cache/qtt/meta",
    ckpt_name="ckpt.pth",
    seed: Optional[int] = None,
    log_freq: int = 50,
):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, ckpt_name)

    metric = "cost"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.train()
    model.to(device)

    loader = MetaDataLoader(metaset, batch_size, 0.1, seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)

    min_loss = float("inf")
    loss_history = deque(maxlen=20)
    start_time = time.time()
    for it in range(1, train_iter + 1):
        optimizer.zero_grad()
        batch = loader.get_batch(metric=metric)
        for key, item in batch.items():
            batch[key] = item.to(device)

        target = batch.pop("target")
        output = model(**batch)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        scheduler.step() if scheduler is not None else None

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
            val_error = 0
            for _ in range(val_iter):
                batch = loader.get_batch(mode="val", metric=metric)
                for key, item in batch.items():
                    batch[key] = item.to(device)
                target = batch.pop("target")
                output = model(**batch)
                loss = torch.nn.functional.l1_loss(output, target)
                val_error += loss.item()

            if val_error < min_loss:
                min_loss = val_error
                torch.save(model.state_dict(), save_path)
                print(f"VAL  [{it}]: Checkpoint saved with val-error: {val_error:.3f}")
            else:
                print(f"VAL  [{it}]: val-error: {val_error:.3f}")

            model.train()

    # Load the model with the best validation error
    print(f"Loading the model with the best validation error: {min_loss:.3f}")
    model.load_state_dict(torch.load(os.path.join(cache_dir, ckpt_name)))

    return model
