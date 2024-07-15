import os

import pandas as pd
from ConfigSpace.read_and_write import json as cs_json

from qtt.configuration import ConfigManager
from qtt.optimizers import QuickOptimizer
from qtt.optimizers.surrogates.dyhpo import DyHPO
from qtt.optimizers.surrogates.estimator import CostEstimator

def get_metatrained_surrogates(name_or_path: str):
    if name_or_path.startswith("mtlbm/"):
        _, version = name_or_path.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(file_path, "pretrained", "mtlbm", version)
    else:
        assert os.path.exists(name_or_path), f"{name_or_path} does not exist."
        root = name_or_path
    
    config_path = os.path.join(root, "space.json")
    config_space = cs_json.read(open(config_path, "r").read())

    cfg_nrm_path = os.path.join(root, "config_norm.csv")
    config_norm = pd.read_csv(cfg_nrm_path, index_col=0)

    mtft_nrm_path = os.path.join(root, "metafeat_norm.csv")
    metafeat_norm = pd.read_csv(mtft_nrm_path, index_col=0)
    
    dyhpo = DyHPO.from_pretrained(root)
    cost_estimator = CostEstimator.from_pretrained(root)

    return dyhpo, cost_estimator, config_space, config_norm, metafeat_norm


def get_optimizer(name_or_path: str):
    if name_or_path.startswith("mtlbm/"):
        _, version = name_or_path.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(file_path, "pretrained", "mtlbm", version)
    else:
        assert os.path.exists(name_or_path), f"{name_or_path} does not exist."
        root = name_or_path

    config_path = os.path.join(root, "space.json")
    config_space = cs_json.read(open(config_path, "r").read())

    norm_path = os.path.join(root, "config_norm.csv")
    config_norm = pd.read_csv(norm_path, index_col=0)
    cm = ConfigManager(config_space, config_norm)
    dyhpo = DyHPO.from_pretrained(root)
    cost_estimator = CostEstimator.from_pretrained(root)

    optimizer = QuickOptimizer(cm, dyhpo, cost_estimator)
    return optimizer
