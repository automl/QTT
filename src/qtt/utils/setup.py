import os

import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from qtt.configuration import ConfigManager
from qtt.optimizers.quick import QuickOptimizer
from qtt.optimizers.surrogates.dyhpo import DyHPO


def get_opt_from_pretrained(path_or_name: str, num_configs: int = 128):
    """
    Load a pretrained optimizer from .

    Parameters
    ----------
    path_or_name : str
        The path to the pretrained model or the name of the pretrained model.
    num_configs : int, default = 128
        The number of candidate configurations to generate.

    Returns
    -------
    QuickOptimizer
    """
    if path_or_name.startswith("mtlbm/"):
        _, version = path_or_name.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(file_path, "..", "pretrained", "mtlbm")
        config_path = os.path.join(root, "mtlbm.json")
        meta_data_path = os.path.join(root, version, "meta_info.csv")
        surrogate_path = os.path.join(root, version)
    else:
        assert os.path.exists(path_or_name), f"{path_or_name} does not exist."
        root = path_or_name
        config_path = os.path.join(root, "mtlbm.json")
        meta_data_path = os.path.join(root, "meta_info.csv")
        surrogate_path = root

    config_space = cs_json.read(open(config_path, "r").read())
    meta_data = pd.read_csv(meta_data_path, index_col=0)
    manager = ConfigManager(config_space, meta_data)
    dyhpo = DyHPO.from_pretrained(surrogate_path)

    optimizer = QuickOptimizer(dyhpo, manager, num_configs)
    return optimizer
