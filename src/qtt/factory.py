import json
import os

import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from qtt.configuration import ConfigManager
from qtt.optimizers import QuickOptimizer
from qtt.optimizers.surrogates.dyhpo import DyHPO
from qtt.optimizers.surrogates.estimator import CostEstimator


def get_opt(name_or_path: str, pretrained: bool = False):
    if pretrained:
        return get_opt_from_pretrained(name_or_path)
    else:
        return get_opt_from_scratch(name_or_path)


def get_opt_from_pretrained(name_or_path: str):
    if name_or_path.startswith("mtlbm/"):
        _, version = name_or_path.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(file_path, "pretrained", "mtlbm", version)
    else:
        assert os.path.exists(name_or_path), f"{name_or_path} does not exist."
        root = name_or_path
        
    config_path = os.path.join(root, "mtlbm.json")
    meta_data_path = os.path.join(root, "meta_info.csv")

    config_space = cs_json.read(open(config_path, "r").read())
    meta_data = pd.read_csv(meta_data_path, index_col=0)
    manager = ConfigManager(config_space, meta_data)
    dyhpo = DyHPO.from_pretrained(root)
    cost_estimator = CostEstimator.from_pretrained(root)

    optimizer = QuickOptimizer(dyhpo, cost_estimator)
    return optimizer, manager


def get_opt_from_scratch(name_or_path: str):
    if name_or_path.startswith("mtlbm/"):
        _, version = name_or_path.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(file_path, "pretrained", "mtlbm", version)
        config_path = os.path.join(root, "mtlbm.json")
        meta_data_path = os.path.join(root, "meta_info.csv")
    else:
        assert os.path.exists(name_or_path), f"{name_or_path} does not exist."
        root = name_or_path
        config_path = os.path.join(root, "mtlbm.json")
        meta_data_path = os.path.join(root, "meta_info.csv")

    config_space = cs_json.read(open(config_path, "r").read())
    meta_data = pd.read_csv(meta_data_path, index_col=0)
    manager = ConfigManager(config_space, meta_data)
    config = json.load(open(os.path.join(root, "config.json"), "r"))
    dyhpo = DyHPO(**config)
    cost_estimator = CostEstimator(**config)

    optimizer = QuickOptimizer(dyhpo, cost_estimator)
    return optimizer, manager
