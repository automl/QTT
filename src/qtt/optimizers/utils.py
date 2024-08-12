from pathlib import Path

from .quick import QuickOptimizer


def get_pretrained_optimizer(path: str) -> QuickOptimizer:
    if path.startswith("mtlbm"):
        _, version = path.split("/")
        optimizer_path = Path(__file__).parent.parent / "pretrained" / "mtlbm" / version
    assert optimizer_path.exists(), f"Optimizer not found at {optimizer_path}"
    return QuickOptimizer.load(optimizer_path)
