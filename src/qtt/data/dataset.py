import json
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch

NUM_HP = [
    "bss_reg",
    "clip_grad",
    "cotuning_reg",
    "cutmix",
    "decay_rate",
    "delta_reg",
    "drop",
    "layer_decay",
    "lr",
    "mixup",
    "mixup_prob",
    "momentum",
    "pct_to_freeze",
    "smoothing",
    "sp_reg",
    "warmup_lr",
    "weight_decay",
]


class MTLBMDataSet:
    def __init__(
        self,
        root: str,
        standardize: bool = True,
        sort_hp: bool = True,
        return_tensor: bool = True,
        *,
        max_budget: int = 50,
    ):
        self.root = root
        self.standardize = standardize
        self.sort_hp = sort_hp
        self.return_tensor = return_tensor
        self.max_budget = max_budget

        self.configs = self._load_configs()
        self.curve_norm = {"cost": 1.0, "perf": 100.0}
        self.metrics = ["cost", "perf"]
        self.curves = self._load_curves()
        self.metafeatures = self._load_meta()
        self.datasets, self.ds_to_exp_ids = self._get_info()

    def _load_configs(self) -> pd.DataFrame:
        path = os.path.join(self.root, "configs.csv")
        df = pd.read_csv(path, index_col=0)

        if self.standardize:
            self.cfg_mean = df.mean()
            self.cfg_std = df.std()
            df[NUM_HP] = df[NUM_HP] - self.cfg_mean[NUM_HP] / self.cfg_std[NUM_HP]

        if self.sort_hp:
            cols = list(df.columns)
            cols.sort()
            models = [col for col in cols if col.startswith("model")]
            others = [col for col in cols if col not in models]
            cols = models + others
            df = df[cols]

        df = df.astype(float)
        return df

    def _load_curves(self):
        curves = {}
        for curve in self.metrics:
            path = os.path.join(self.root, f"{curve}.csv")
            data = pd.read_csv(path, index_col=0)
            curves[curve] = data
        return curves

    def _load_meta(self) -> pd.DataFrame:
        path = os.path.join(self.root, "meta.csv")
        metafeatures = pd.read_csv(path, index_col=0)
        metafeatures = metafeatures / 10000  # TODO: standardize
        # self.metafeat_mean = metafeatures.mean()
        # self.metafeat_std = metafeatures.std()
        # metafeatures = (metafeatures - self.metafeat_mean) / self.metafeat_std
        # metafeatures[metafeatures.isnull()] = 0
        return metafeatures

    def _get_info(self) -> Tuple[List[str], dict[str, List[int]]]:
        path = os.path.join(self.root, "ds_to_idx.json")
        ds_to_idx = json.load(open(path))
        datasets = list(ds_to_idx.keys())
        return datasets, ds_to_idx

    def __len__(self):
        return len(self.configs)

    def get_batch(
        self,
        batch_size: int,
        metric: str = "perf",
        dataset: Optional[str] = None,
    ) -> dict:
        if dataset is None:
            dataset = random.choice(self.datasets)
        assert dataset in self.datasets, f"{dataset} not found in the MetaSet"
        assert metric in self.metrics, f"{metric} not found in the MetaSet"

        exp_idx = random.sample(self.ds_to_exp_ids[dataset], batch_size)
        exp_idx = list(map(int, exp_idx))

        curve, target, budget = [], [], []

        for idx in exp_idx:
            crv = self.curves[metric].loc[idx].values
            crv = crv[~pd.isnull(crv)].tolist()
            bdgt = random.randint(1, self.max_budget)
            bdgt = min(bdgt, len(crv))
            crv = crv[:bdgt]
            trgt = crv[-1]
            crv = crv[:-1]
            crv = crv + [0] * (self.max_budget - len(crv))
            curve.append(crv)
            target.append(trgt)
            budget.append(bdgt)

        config = self.configs.loc[exp_idx].values
        metafeat = self.metafeatures.loc[exp_idx].values

        curve_norm = self.curve_norm[metric]
        if self.return_tensor:
            config = torch.tensor(config, dtype=torch.float32)
            curve = torch.tensor(curve, dtype=torch.float32) / curve_norm
            target = torch.tensor(target, dtype=torch.float32) / curve_norm
            budget = torch.tensor(budget, dtype=torch.float32) / self.max_budget
            metafeat = torch.tensor(metafeat, dtype=torch.float32)

        batch = dict(
            config=config,
            curve=curve,
            budget=budget,
            target=target,
            metafeat=metafeat,
        )
        return batch
    
    @property
    def hyperparameter_names(self) -> List[str]:
        return list(self.configs.columns)

    @property
    def num_hps(self) -> int:
        return len(self.configs.columns)

    def get_num_datasets(self) -> int:
        return len(self.datasets)

    def get_datasets(self) -> List[str]:
        return self.datasets

    def get_hp_candidates(self):
        return self.configs.values
    
    def get_meta_data(self):
        mean = self.cfg_mean
        std = self.cfg_std

        # Set mean and std of categorical and non-numerical hyperparameters to 0
        NON_NUM_HP = [col for col in self.configs.columns if col not in NUM_HP]
        mean[NON_NUM_HP] = 0
        std[NON_NUM_HP] = 0

        mean = mean[self.configs.columns]
        std = std[self.configs.columns]
        return pd.DataFrame([mean, std], index=["mean", "std"])

    def save_data_info(self, path: str = "."):
        df = self.get_meta_data()
        save_path = os.path.join(path, "meta_info.csv")
        df.to_csv(save_path)