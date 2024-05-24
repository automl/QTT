import logging
import os
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from qtt.data.loader import MetaDataLoader
from qtt.data.dataset import MTLBMDataSet
from qtt.optimizers.surrogates.surrogate import Surrogate


logger = logging.getLogger("metatrain")


def meta_train_surrogate(
    surrogate: Surrogate,
    metaset: MTLBMDataSet,
    batch_size: int = 32,
    lr: float = 1e-4,
    train_iter: int = 10000,
    val_iter: int = 50,
    val_freq: int = 20,
    use_scheduler: bool = True,
    device="auto",
    cache_dir="~/.cache/qtt/meta",
    ckpt_name="ckpt.pth",
    seed: Optional[int] = None,
):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, ckpt_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(device)
    surrogate.to(device)
    surrogate.train()

    loader = MetaDataLoader(metaset, batch_size, seed)
    for metric in ("perf", "cost"):
        optimizer = torch.optim.Adam(surrogate.parameters(), lr)
        scheduler = None
        if use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)

        min_loss = float("inf")
        for it in range(1, train_iter + 1):
            optimizer.zero_grad()
            batch = loader.get_batch(metric=metric)
            for key, item in batch.items():
                batch[key] = item.to(device)

            if metric == "perf":
                loss = surrogate(**batch)
                loss.backward()
                optimizer.step()
            else:
                target = batch.pop("target")
                logits = surrogate.cost_predictor(**batch)
                loss = torch.nn.functional.mse_loss(logits.reshape(target.shape), target)
                loss.backward()
                optimizer.step()

            scheduler.step() if scheduler is not None else None

            if not it % val_freq:
                surrogate.eval()
                val_loss = validate(surrogate, metric, val_iter, device, loader)

                logger.debug(f"Step {it}: val_loss={val_loss:.4f}")
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(surrogate.state_dict(), save_path)
                    logger.info(f"Iter {it}: Checkpoint saved with Val-loss {val_loss:.4f}")
                    print(f"Iter {it}: Checkpoint saved with Val-loss {val_loss:.4f}")
                surrogate.train()

        # Load the model with the best validation error
        logger.info(f"Loading the model with the best validation error: {min_loss:.4f}")
        surrogate.load_state_dict(torch.load(os.path.join(cache_dir, ckpt_name)))
    
    return surrogate


def validate(surrogate, metric, val_iter, device, loader):
    val_loss = 0
    for _ in range(val_iter):
        if metric == "perf":
            batch_train = loader.get_batch(mode="val")
            batch_test = loader.get_batch(mode="val")
            for key, item in batch_train.items():
                batch_train[key] = item.to(device)
            for key, item in batch_test.items():
                batch_test[key] = item.to(device)

            target = batch_test.pop("target")

            mean, _, _ = surrogate.predict_pipeline(batch_train, batch_test)
            loss = torch.nn.functional.mse_loss(mean, target)
        else:
            batch = loader.get_batch(mode="val", metric=metric)
            for key, item in batch.items():
                batch[key] = item.to(device)
            target = batch.pop("target")

            logits = surrogate.cost_predictor(**batch)
            logits = logits.reshape(target.shape)
            loss = torch.nn.functional.mse_loss(logits, target)

        val_loss += loss.item()
    val_loss /= val_iter
    return val_loss
