import os
from ..optimizers import QuickOptimizer


def load_pretrained_optimizer(name: str):
    """Get a pretrained optimizer.

    Args:
        name (str):
            Name of the pretrained optimizer.

    Returns:
        Optimizer: A pretrained optimizer.
    """
    path = os.path.dirname(__file__)
    path = os.path.join(path, name)
    if os.path.exists(path):
        path = os.path.join(path, "optimizer")
    else:
        raise ValueError(f"Pretrained optimizer '{name}' not found.")
    return QuickOptimizer.load(str(path))


__all__ = [
    "load_pretrained_optimizer",
]