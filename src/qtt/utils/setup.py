import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_outputdir(path, create_dir=True, warn_if_exist=True, path_suffix=None):
    if path is not None:
        assert isinstance(
            path, str
        ), f"Only str is supported for path, got {path} of type {type(path)}."

    if path_suffix is None:
        path_suffix = ""
    if path is not None:
        path = os.path.join(path, path_suffix)
    if path is None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        path = os.path.join("qtt", timestamp, path_suffix)
    if create_dir:
        try:
            os.makedirs(path, exist_ok=False)
            logger.info(f"Created directory: {path}")
        except FileExistsError:
            logger.warning(f"'{path}' already exists! This may overwrite old data.")
    elif warn_if_exist:
        if os.path.isdir(path):
            logger.warning(f"'{path}' already exists! This may overwrite old data.")
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    return path


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
